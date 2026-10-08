"""Evaluate Phase 2 Method 3 without selecting features on official test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.audit_data import assert_clean_report, build_report
from src.evaluate_models import RUL_BIN_EDGES, RUL_BIN_LABELS
from src.prepare_data import (
    ConditionAwareTransformer,
    add_training_rul,
    condition_aware_feature_columns,
    final_test_observations,
    load_test_rul,
    load_trajectory,
    split_development_and_calibration,
)
from src.temporal_features import (
    TEMPORAL_STATISTICS,
    TEMPORAL_WINDOW,
    add_temporal_features,
    temporal_feature_columns,
)


RANDOM_SEED = 42
PHASE2_SEED = 2026
VALIDATION_FRACTION = 0.20
VALIDATION_ENDPOINTS_PER_ENGINE = 5


def make_point_model() -> HistGradientBoostingRegressor:
    """Match the fixed Phase 1 median-model configuration."""
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.50,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def split_phase2_training_validation(
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve complete engines for Method 3 design validation."""
    unit_ids = np.sort(development["unit_id"].unique())
    training_ids, validation_ids = train_test_split(
        unit_ids,
        test_size=VALIDATION_FRACTION,
        random_state=PHASE2_SEED,
        shuffle=True,
    )
    training = development[development["unit_id"].isin(training_ids)].copy()
    validation = development[development["unit_id"].isin(validation_ids)].copy()
    if set(training["unit_id"]) & set(validation["unit_id"]):
        raise RuntimeError("Engine leakage detected in the Phase 2 validation split.")
    return training, validation


def engine_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each complete engine equal total fitting weight."""
    counts = frame.groupby("unit_id")["unit_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights * (len(weights) / weights.sum())


def sample_validation_endpoints(
    validation: pd.DataFrame,
    endpoints_per_engine: int = VALIDATION_ENDPOINTS_PER_ENGINE,
    random_seed: int = PHASE2_SEED,
) -> pd.DataFrame:
    """Sample the same fixed number of observed cycles from every validation engine."""
    if endpoints_per_engine < 1:
        raise ValueError("endpoints_per_engine must be at least 1.")
    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    for unit_id, unit_frame in validation.groupby("unit_id", sort=True):
        if len(unit_frame) < endpoints_per_engine:
            raise ValueError(
                f"Unit {unit_id} has fewer rows than the requested validation endpoints."
            )
        selected.extend(
            int(index)
            for index in rng.choice(
                unit_frame.index.to_numpy(), size=endpoints_per_engine, replace=False
            )
        )
    return validation.loc[selected].sort_values(["unit_id", "cycle"]).copy()


def prediction_frame(
    arm: str,
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    output = frame[["unit_id", "cycle", target_column]].copy()
    output = output.rename(columns={target_column: "true_rul"})
    output["predicted_rul"] = model.predict(frame[feature_columns])
    output["error"] = output["predicted_rul"] - output["true_rul"]
    output["absolute_error"] = output["error"].abs()
    output["arm"] = arm
    output["rul_band"] = pd.cut(
        output["true_rul"],
        bins=RUL_BIN_EDGES,
        labels=RUL_BIN_LABELS,
        include_lowest=True,
    )
    return output


def metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "engines": int(frame["unit_id"].nunique()),
        "mae": float(mean_absolute_error(frame["true_rul"], frame["predicted_rul"])),
        "rmse": float(mean_squared_error(frame["true_rul"], frame["predicted_rul"]) ** 0.5),
        "mean_error": float(frame["error"].mean()),
    }


def paired_engine_bootstrap(
    predictions: pd.DataFrame,
    repetitions: int = 10000,
) -> dict[str, Any]:
    """Bootstrap the paired engine-average MAE improvement."""
    engine_errors = (
        predictions.groupby(["unit_id", "arm"], as_index=False)["absolute_error"].mean()
        .pivot(index="unit_id", columns="arm", values="absolute_error")
    )
    differences = (
        engine_errors["phase2_snapshot"] - engine_errors["phase2_engineered_history"]
    ).to_numpy()
    rng = np.random.default_rng(PHASE2_SEED)
    bootstrap = np.empty(repetitions)
    for index in range(repetitions):
        bootstrap[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return {
        "engines": int(len(differences)),
        "mean_mae_improvement": float(differences.mean()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_95_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "fraction_of_engines_improved": float((differences > 0).mean()),
    }


def by_rul_band(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band, band_frame in predictions.groupby("rul_band", observed=False):
        if band_frame.empty:
            continue
        arms = {
            arm: frame["absolute_error"].mean()
            for arm, frame in band_frame.groupby("arm", observed=False)
        }
        rows.append(
            {
                "rul_band": str(band),
                "endpoints_per_arm": int(
                    len(band_frame[band_frame["arm"] == "phase2_engineered_history"])
                ),
                "snapshot_mae": float(arms["phase2_snapshot"]),
                "engineered_history_mae": float(arms["phase2_engineered_history"]),
                "mae_improvement": float(
                    arms["phase2_snapshot"] - arms["phase2_engineered_history"]
                ),
            }
        )
    return rows


def fit_and_compare(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    training_target_column: str,
    evaluation_target_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    snapshot_columns = condition_aware_feature_columns()
    history_columns = [*snapshot_columns, *temporal_feature_columns()]
    weights = engine_balanced_weights(training)

    snapshot_model = make_point_model().fit(
        training[snapshot_columns], training[training_target_column], sample_weight=weights
    )
    history_model = make_point_model().fit(
        training[history_columns], training[training_target_column], sample_weight=weights
    )
    predictions = pd.concat(
        [
            prediction_frame(
                "phase2_snapshot",
                snapshot_model,
                evaluation,
                snapshot_columns,
                evaluation_target_column,
            ),
            prediction_frame(
                "phase2_engineered_history",
                history_model,
                evaluation,
                history_columns,
                evaluation_target_column,
            ),
        ],
        ignore_index=True,
    )
    report = {
        "arms": {
            arm: {
                "feature_count": len(
                    snapshot_columns if arm == "phase2_snapshot" else history_columns
                ),
                **metric_summary(frame),
            }
            for arm, frame in predictions.groupby("arm", observed=False)
        },
        "paired_engine_bootstrap": paired_engine_bootstrap(predictions),
        "by_true_rul_band": by_rul_band(predictions),
    }
    return report, predictions


def run_method3(raw_directory: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run locked Method 3 validation and the labelled exploratory test comparison."""
    assert_clean_report(build_report(raw_directory))
    train = add_training_rul(load_trajectory(raw_directory / "train_FD002.txt"))
    test = load_trajectory(raw_directory / "test_FD002.txt")
    test_rul = load_test_rul(raw_directory / "RUL_FD002.txt", test)
    development, calibration = split_development_and_calibration(train)
    phase2_training, phase2_validation = split_phase2_training_validation(development)

    validation_transformer = ConditionAwareTransformer().fit(phase2_training)
    validation_training = add_temporal_features(
        validation_transformer.transform(phase2_training)
    )
    validation_pool = add_temporal_features(
        validation_transformer.transform(phase2_validation)
    )
    validation_endpoints = sample_validation_endpoints(validation_pool)
    validation_report, validation_predictions = fit_and_compare(
        validation_training, validation_endpoints, "rul", "rul"
    )

    final_transformer = ConditionAwareTransformer().fit(development)
    final_training = add_temporal_features(final_transformer.transform(development))
    full_test_transformed = add_temporal_features(final_transformer.transform(test))
    test_endpoints = final_test_observations(full_test_transformed, test_rul)
    test_report, test_predictions = fit_and_compare(
        final_training, test_endpoints, "rul", "true_rul"
    )

    report = {
        "status": (
            "Exploratory Phase 2 analysis designed after Phase 1 official-test results "
            "were inspected."
        ),
        "method": {
            "name": "Method 3: engineered trailing-history features",
            "window_observations": TEMPORAL_WINDOW,
            "source_signals": "21 anonymous within-regime sensor deviations",
            "statistics_per_signal": list(TEMPORAL_STATISTICS),
            "rul_target": "uncapped remaining observed cycles",
            "training_weighting": "equal total sample weight per engine",
        },
        "splits": {
            "phase1_development_engines": int(development["unit_id"].nunique()),
            "phase1_untouched_calibration_engines": int(calibration["unit_id"].nunique()),
            "method3_training_engines": int(phase2_training["unit_id"].nunique()),
            "method3_validation_engines": int(phase2_validation["unit_id"].nunique()),
            "validation_endpoints_per_engine": VALIDATION_ENDPOINTS_PER_ENGINE,
            "official_test_engines": int(test_endpoints["unit_id"].nunique()),
        },
        "development_validation": validation_report,
        "official_test_exploratory": test_report,
    }
    predictions = pd.concat(
        [
            validation_predictions.assign(evaluation_split="development_validation"),
            test_predictions.assign(evaluation_split="official_test_exploratory"),
        ],
        ignore_index=True,
    )
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 Method 3 on FD002.")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/phase2_method3_results.json")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/processed/phase2_method3_predictions.csv"),
    )
    args = parser.parse_args()

    report, predictions = run_method3(args.raw_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(args.predictions_output, index=False)
    print(f"Wrote Method 3 results to {args.output}")
    print(f"Wrote Method 3 predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()

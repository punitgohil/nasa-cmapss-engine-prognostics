"""Compare global and condition-aware RUL models on official FD002 test endpoints.

All transformations and models are fit without accessing official test labels.
The script reports results; it does not make maintenance or deployment claims.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.audit_data import assert_clean_report, build_report
from src.prepare_data import (
    ConditionAwareTransformer,
    add_training_rul,
    condition_aware_feature_columns,
    final_test_observations,
    global_feature_columns,
    load_test_rul,
    load_trajectory,
    split_development_and_calibration,
)


CENTRAL_QUANTILE = 0.50
INTERVAL_LEVELS: dict[str, tuple[float, float, float]] = {
    "80": (0.80, 0.10, 0.90),
    "90": (0.90, 0.05, 0.95),
    "95": (0.95, 0.025, 0.975),
}
RANDOM_SEED = 42
RUL_BIN_EDGES = [-np.inf, 25, 50, 100, np.inf]
RUL_BIN_LABELS = ["<=25", "26-50", "51-100", ">100"]


@dataclass
class MultiIntervalModelSet:
    models: dict[float, HistGradientBoostingRegressor]
    conformal_adjustments: dict[str, float]


def _make_model(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def conformal_adjustment(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.90,
) -> float:
    """Return the finite-sample split-conformal expansion for a target coverage."""
    if not (len(y_true) == len(lower) == len(upper)) or len(y_true) == 0:
        raise ValueError("Calibration targets and interval bounds must have equal non-zero lengths.")
    if not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must be between 0 and 1.")
    scores = np.maximum(lower - y_true, y_true - upper)
    corrected_quantile = min(1.0, ((len(scores) + 1) * nominal_coverage) / len(scores))
    return float(np.quantile(scores, corrected_quantile, method="higher"))


def sample_calibration_endpoints(
    calibration: pd.DataFrame, random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Select one deterministic pseudo-endpoint per held-out calibration unit.

    Using one row per unit avoids treating correlated rows from the same engine
    trajectory as independent calibration examples. The selected cycle is drawn
    uniformly from each unit's observed trajectory with a fixed seed.
    """
    rng = np.random.default_rng(random_seed)
    selected_indexes: list[int] = []
    for _, unit_frame in calibration.groupby("unit_id", sort=True):
        selected_indexes.append(int(rng.choice(unit_frame.index.to_numpy())))
    return calibration.loc[selected_indexes].sort_values("unit_id").copy()


def fit_quantile_models(
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
) -> MultiIntervalModelSet:
    """Fit required quantiles once and calibrate each interval independently."""
    x_development = development[feature_columns]
    y_development = development["rul"]
    x_calibration = calibration[feature_columns]
    y_calibration = calibration["rul"].to_numpy()

    required_quantiles = {CENTRAL_QUANTILE}
    for _, lower_quantile, upper_quantile in INTERVAL_LEVELS.values():
        required_quantiles.update((lower_quantile, upper_quantile))

    models = {
        quantile: _make_model(quantile).fit(x_development, y_development)
        for quantile in sorted(required_quantiles)
    }
    adjustments: dict[str, float] = {}
    for label, (coverage, lower_quantile, upper_quantile) in INTERVAL_LEVELS.items():
        adjustments[label] = conformal_adjustment(
            y_calibration,
            models[lower_quantile].predict(x_calibration),
            models[upper_quantile].predict(x_calibration),
            nominal_coverage=coverage,
        )
    return MultiIntervalModelSet(models=models, conformal_adjustments=adjustments)


def predict_intervals(model_set: MultiIntervalModelSet, features: pd.DataFrame) -> pd.DataFrame:
    """Return the central estimate and all conformally adjusted intervals."""
    predictions: dict[str, np.ndarray] = {
        "predicted_rul": model_set.models[CENTRAL_QUANTILE].predict(features)
    }
    for label, (_, lower_quantile, upper_quantile) in INTERVAL_LEVELS.items():
        adjustment = model_set.conformal_adjustments[label]
        predictions[f"lower_{label}"] = model_set.models[lower_quantile].predict(features) - adjustment
        predictions[f"upper_{label}"] = model_set.models[upper_quantile].predict(features) + adjustment
    return enforce_interval_nesting(pd.DataFrame(predictions, index=features.index))


def enforce_interval_nesting(predictions: pd.DataFrame) -> pd.DataFrame:
    """Ensure higher-coverage intervals contain every lower-coverage interval."""
    nested = predictions.copy()
    for inner, outer in (("80", "90"), ("90", "95")):
        nested[f"lower_{outer}"] = np.minimum(nested[f"lower_{outer}"], nested[f"lower_{inner}"])
        nested[f"upper_{outer}"] = np.maximum(nested[f"upper_{outer}"], nested[f"upper_{inner}"])
    return nested


def point_metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["true_rul"].to_numpy()
    central = frame["predicted_rul"].to_numpy()
    return {
        "n": int(len(frame)),
        "mae": float(mean_absolute_error(y_true, central)),
        "rmse": float(mean_squared_error(y_true, central) ** 0.5),
        "mean_error": float((central - y_true).mean()),
    }


def interval_metric_summary(frame: pd.DataFrame, label: str) -> dict[str, float]:
    lower = frame[f"lower_{label}"]
    upper = frame[f"upper_{label}"]
    covered = (frame["true_rul"] >= lower) & (frame["true_rul"] <= upper)
    return {
        "nominal_coverage": INTERVAL_LEVELS[label][0],
        "empirical_coverage": float(covered.mean()),
        "mean_interval_width": float((upper - lower).mean()),
    }


def grouped_metrics(frame: pd.DataFrame, group_column: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for group, group_frame in frame.groupby(group_column, observed=False):
        if len(group_frame) == 0:
            continue
        results[str(group)] = {
            **point_metric_summary(group_frame),
            "interval_90": interval_metric_summary(group_frame, "90"),
        }
    return results


def evaluate_arm(
    name: str,
    development: pd.DataFrame,
    calibration: pd.DataFrame,
    test_endpoints: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    model_set = fit_quantile_models(development, calibration, feature_columns)
    predictions = predict_intervals(model_set, test_endpoints[feature_columns])
    evaluation = pd.concat([test_endpoints[["unit_id", "true_rul", "regime"]], predictions], axis=1)
    evaluation["arm"] = name
    evaluation["error"] = evaluation["predicted_rul"] - evaluation["true_rul"]
    evaluation["absolute_error"] = evaluation["error"].abs()
    evaluation["rul_band"] = pd.cut(
        evaluation["true_rul"], bins=RUL_BIN_EDGES, labels=RUL_BIN_LABELS, include_lowest=True
    )
    report = {
        "arm": name,
        "feature_count": len(feature_columns),
        "conformal_adjustments": model_set.conformal_adjustments,
        "overall": point_metric_summary(evaluation),
        "intervals": {
            label: interval_metric_summary(evaluation, label) for label in INTERVAL_LEVELS
        },
        "by_regime": grouped_metrics(evaluation, "regime"),
        "by_true_rul_band": grouped_metrics(evaluation, "rul_band"),
    }
    return report, evaluation


def run_evaluation(raw_directory: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the locked comparison and return metrics plus endpoint predictions."""
    assert_clean_report(build_report(raw_directory))
    train = add_training_rul(load_trajectory(raw_directory / "train_FD002.txt"))
    test = load_trajectory(raw_directory / "test_FD002.txt")
    test_rul = load_test_rul(raw_directory / "RUL_FD002.txt", test)
    development, calibration = split_development_and_calibration(train)
    calibration_endpoints = sample_calibration_endpoints(calibration)

    transformer = ConditionAwareTransformer().fit(development)
    development_condition = transformer.transform(development)
    calibration_condition = transformer.transform(calibration)
    calibration_condition_endpoints = calibration_condition.loc[calibration_endpoints.index].copy()
    test_condition = transformer.transform(final_test_observations(test, test_rul))

    global_report, global_predictions = evaluate_arm(
        "global_baseline",
        development,
        calibration_endpoints,
        test_condition,
        global_feature_columns(),
    )
    condition_report, condition_predictions = evaluate_arm(
        "condition_aware",
        development_condition,
        calibration_condition_endpoints,
        test_condition,
        condition_aware_feature_columns(),
    )
    quantiles = {CENTRAL_QUANTILE}
    quantiles.update(value for values in INTERVAL_LEVELS.values() for value in values[1:])
    report = {
        "scope": "NASA C-MAPSS FD002 simulated benchmark; official test labels used only for final evaluation.",
        "model": {
            "family": "HistGradientBoostingRegressor",
            "quantiles": sorted(quantiles),
            "nominal_interval_coverages": [values[0] for values in INTERVAL_LEVELS.values()],
            "calibration_method": (
                "split conformal adjustment using one deterministic uniformly sampled "
                "pseudo-endpoint per held-out calibration trajectory"
            ),
        },
        "split": {
            "development_units": int(development["unit_id"].nunique()),
            "calibration_units": int(calibration["unit_id"].nunique()),
            "calibration_endpoint_rows": int(len(calibration_endpoints)),
            "calibration_endpoint_sampling": (
                "one uniformly sampled observed cycle per calibration unit; random seed 42"
            ),
            "calibration_endpoint_rul_summary": {
                "minimum": float(calibration_endpoints["rul"].min()),
                "median": float(calibration_endpoints["rul"].median()),
                "maximum": float(calibration_endpoints["rul"].max()),
            },
            "official_test_units": int(test_condition["unit_id"].nunique()),
        },
        "arms": [global_report, condition_report],
    }
    predictions = pd.concat([global_predictions, condition_predictions], ignore_index=True)
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate global and condition-aware FD002 RUL models.")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/evaluation_results.json"))
    parser.add_argument(
        "--predictions-output", type=Path, default=Path("data/processed/engine_predictions.csv")
    )
    args = parser.parse_args()

    report, predictions = run_evaluation(args.raw_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(args.predictions_output, index=False)
    print(f"Wrote evaluation results to {args.output}")
    print(f"Wrote engine predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()

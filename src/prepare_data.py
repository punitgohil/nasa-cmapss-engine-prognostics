"""Prepare NASA C-MAPSS FD002 data without leaking held-out information.

This module creates the row-level training target, holds out complete engine
trajectories for calibration, and learns operating regimes from development data
only. It deliberately does not assign physical meanings to anonymous columns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.audit_data import COLUMN_NAMES, assert_clean_report, build_report


OPERATION_SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{number}" for number in range(1, 22)]
RANDOM_SEED = 42
CALIBRATION_FRACTION = 0.20
REGIME_COUNT = 6


def load_trajectory(path: Path) -> pd.DataFrame:
    """Load one headerless, whitespace-separated C-MAPSS trajectory file."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python")
    if len(frame.columns) != len(COLUMN_NAMES):
        raise ValueError(f"Expected {len(COLUMN_NAMES)} columns in {path.name}.")
    if frame.isna().any().any():
        raise ValueError(f"Missing values found in {path.name}; run the audit before preparation.")

    frame["unit_id"] = frame["unit_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame


def load_test_rul(path: Path, test_frame: pd.DataFrame) -> pd.DataFrame:
    """Align official RUL values to final test observations in unit-ID order."""
    values = pd.read_csv(path, sep=r"\s+", header=None, names=["true_rul"], engine="python")
    unit_ids = np.sort(test_frame["unit_id"].unique())
    if len(values) != len(unit_ids):
        raise ValueError(
            f"RUL file has {len(values)} values but test data has {len(unit_ids)} distinct units."
        )
    expected_unit_ids = np.arange(1, len(unit_ids) + 1)
    if not np.array_equal(unit_ids, expected_unit_ids):
        raise ValueError(
            "The headerless RUL vector can only be aligned safely when test unit IDs "
            "are contiguous and start at 1."
        )
    if (values["true_rul"] < 0).any() or not np.allclose(
        values["true_rul"], values["true_rul"].round()
    ):
        raise ValueError("Official RUL values must be non-negative integer cycle counts.")
    values["unit_id"] = unit_ids
    values["true_rul"] = values["true_rul"].astype(float)
    return values[["unit_id", "true_rul"]]


def add_training_rul(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Derive RUL from each training unit's observed run-to-failure endpoint."""
    result = train_frame.copy()
    final_cycle = result.groupby("unit_id", sort=False)["cycle"].transform("max")
    result["rul"] = (final_cycle - result["cycle"]).astype(float)
    return result


def final_test_observations(test_frame: pd.DataFrame, rul_frame: pd.DataFrame) -> pd.DataFrame:
    """Return one labelled endpoint record per official test unit."""
    endpoints = test_frame.loc[test_frame.groupby("unit_id", sort=False)["cycle"].idxmax()].copy()
    endpoints = endpoints.merge(rul_frame, on="unit_id", how="left", validate="one_to_one")
    if len(endpoints) != test_frame["unit_id"].nunique():
        raise ValueError("Test endpoint extraction did not return exactly one row per unit.")
    return endpoints.sort_values("unit_id").reset_index(drop=True)


def split_development_and_calibration(
    labelled_train_frame: pd.DataFrame,
    calibration_fraction: float = CALIBRATION_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete units deterministically; individual rows are never split."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    unit_ids = np.sort(labelled_train_frame["unit_id"].unique())
    development_ids, calibration_ids = train_test_split(
        unit_ids, test_size=calibration_fraction, random_state=random_seed, shuffle=True
    )
    development = labelled_train_frame[labelled_train_frame["unit_id"].isin(development_ids)].copy()
    calibration = labelled_train_frame[labelled_train_frame["unit_id"].isin(calibration_ids)].copy()

    if set(development["unit_id"]) & set(calibration["unit_id"]):
        raise RuntimeError("Unit leakage detected between development and calibration sets.")
    return development, calibration


@dataclass
class ConditionAwareTransformer:
    """Learn six operating regimes and development-only sensor reference means."""

    regime_count: int = REGIME_COUNT
    random_seed: int = RANDOM_SEED

    def fit(self, development_frame: pd.DataFrame) -> "ConditionAwareTransformer":
        if len(development_frame) < self.regime_count:
            raise ValueError("Development data has fewer rows than requested regimes.")

        self.setting_scaler_ = StandardScaler().fit(development_frame[OPERATION_SETTING_COLUMNS])
        scaled_settings = self.setting_scaler_.transform(development_frame[OPERATION_SETTING_COLUMNS])
        self.regime_model_ = KMeans(
            n_clusters=self.regime_count, random_state=self.random_seed, n_init=20
        ).fit(scaled_settings)
        development_regimes = self._predict_regimes(development_frame)
        grouped = development_frame.assign(regime=development_regimes).groupby("regime")[SENSOR_COLUMNS].mean()
        self.sensor_means_by_regime_ = grouped.reindex(range(self.regime_count))
        if self.sensor_means_by_regime_.isna().any().any():
            raise RuntimeError("A learned regime has no development observations.")
        return self

    def _predict_regimes(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "setting_scaler_") or not hasattr(self, "regime_model_"):
            raise RuntimeError("Fit the transformer using development data before transforming data.")
        scaled_settings = self.setting_scaler_.transform(frame[OPERATION_SETTING_COLUMNS])
        return self.regime_model_.predict(scaled_settings)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create one-hot regimes and within-regime anonymous sensor deviations."""
        regimes = self._predict_regimes(frame)
        transformed = frame.copy()
        transformed["regime"] = regimes

        regime_indicators = pd.get_dummies(transformed["regime"], prefix="regime", dtype=float)
        regime_indicators = regime_indicators.reindex(
            columns=[f"regime_{index}" for index in range(self.regime_count)], fill_value=0.0
        )
        transformed = pd.concat([transformed, regime_indicators], axis=1)

        reference_means = self.sensor_means_by_regime_.loc[regimes].reset_index(drop=True)
        reference_means.index = transformed.index
        for sensor in SENSOR_COLUMNS:
            transformed[f"{sensor}_deviation"] = transformed[sensor] - reference_means[sensor]
        return transformed


def global_feature_columns() -> list[str]:
    """Features available at a row's prediction time; unit ID is never a feature."""
    return ["cycle", *OPERATION_SETTING_COLUMNS, *SENSOR_COLUMNS]


def condition_aware_feature_columns() -> list[str]:
    return [
        *global_feature_columns(),
        *[f"regime_{index}" for index in range(REGIME_COUNT)],
        *[f"{sensor}_deviation" for sensor in SENSOR_COLUMNS],
    ]


def build_preparation_summary(raw_directory: Path) -> dict[str, int | float]:
    """Run data preparation and return only non-model metadata for verification."""
    assert_clean_report(build_report(raw_directory))
    train = add_training_rul(load_trajectory(raw_directory / "train_FD002.txt"))
    test = load_trajectory(raw_directory / "test_FD002.txt")
    test_rul = load_test_rul(raw_directory / "RUL_FD002.txt", test)
    development, calibration = split_development_and_calibration(train)
    transformer = ConditionAwareTransformer().fit(development)
    calibration_transformed = transformer.transform(calibration)
    test_endpoints = final_test_observations(test, test_rul)
    test_transformed = transformer.transform(test_endpoints)

    return {
        "train_rows": int(len(train)),
        "development_rows": int(len(development)),
        "calibration_rows": int(len(calibration)),
        "development_units": int(development["unit_id"].nunique()),
        "calibration_units": int(calibration["unit_id"].nunique()),
        "official_test_endpoint_rows": int(len(test_endpoints)),
        "regime_count": REGIME_COUNT,
        "calibration_regimes_observed": int(calibration_transformed["regime"].nunique()),
        "official_test_regimes_observed": int(test_transformed["regime"].nunique()),
        "minimum_training_rul": float(train["rul"].min()),
        "maximum_training_rul": float(train["rul"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FD002 data without test-data leakage.")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/preparation_summary.json"))
    args = parser.parse_args()

    summary = build_preparation_summary(args.raw_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote preparation summary to {args.output}")


if __name__ == "__main__":
    main()

"""Leakage-safe trailing trajectory features for Phase 2 Method 3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prepare_data import SENSOR_COLUMNS


TEMPORAL_WINDOW = 30
RECENT_LAG = 5
TEMPORAL_STATISTICS = ("mean", "std", "slope", "window_change", "recent_change")


def temporal_source_columns() -> list[str]:
    """Use anonymous, condition-normalized sensors as the historical signals."""
    return [f"{sensor}_deviation" for sensor in SENSOR_COLUMNS]


def temporal_feature_columns() -> list[str]:
    columns = ["history_observations"]
    columns.extend(
        f"{source}_history_{statistic}"
        for source in temporal_source_columns()
        for statistic in TEMPORAL_STATISTICS
    )
    return columns


def _unit_temporal_features(
    unit_frame: pd.DataFrame,
    source_columns: list[str],
    window_size: int,
    recent_lag: int,
) -> pd.DataFrame:
    """Calculate trailing features for one cycle-sorted engine trajectory."""
    ordered = unit_frame.sort_values("cycle")
    cycles = ordered["cycle"].to_numpy(dtype=float)
    values = ordered[source_columns].to_numpy(dtype=float)
    row_count, _ = values.shape

    end = np.arange(row_count)
    start = np.maximum(0, end - window_size + 1)
    observations = end - start + 1

    def prefix(matrix: np.ndarray) -> np.ndarray:
        return np.vstack([np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)])

    value_prefix = prefix(values)
    squared_value_prefix = prefix(values**2)
    sums = value_prefix[end + 1] - value_prefix[start]
    squared_sums = squared_value_prefix[end + 1] - squared_value_prefix[start]
    cycle_matrix = np.repeat(cycles[:, None], values.shape[1], axis=1)
    cycle_prefix = prefix(cycle_matrix)
    cycle_squared_prefix = prefix(cycle_matrix**2)
    cross_prefix = prefix(cycle_matrix * values)
    cycle_sums = cycle_prefix[end + 1] - cycle_prefix[start]
    cycle_squared_sums = cycle_squared_prefix[end + 1] - cycle_squared_prefix[start]
    cross_sums = cross_prefix[end + 1] - cross_prefix[start]

    count_matrix = observations[:, None].astype(float)
    means = sums / count_matrix
    variances = np.maximum(0.0, squared_sums / count_matrix - means**2)
    standard_deviations = np.sqrt(variances)

    slope_denominator = count_matrix * cycle_squared_sums - cycle_sums**2
    slope_numerator = count_matrix * cross_sums - cycle_sums * sums
    slopes = np.divide(
        slope_numerator,
        slope_denominator,
        out=np.zeros_like(slope_numerator),
        where=slope_denominator > 0,
    )

    window_changes = values - values[start]
    recent_start = np.maximum(start, end - recent_lag)
    recent_changes = values - values[recent_start]

    output_values: dict[str, np.ndarray] = {
        "history_observations": observations.astype(float)
    }
    matrices = {
        "mean": means,
        "std": standard_deviations,
        "slope": slopes,
        "window_change": window_changes,
        "recent_change": recent_changes,
    }
    for column_index, source in enumerate(source_columns):
        for statistic, matrix in matrices.items():
            output_values[f"{source}_history_{statistic}"] = matrix[:, column_index]
    return pd.DataFrame(output_values, index=ordered.index)


def add_temporal_features(
    frame: pd.DataFrame,
    window_size: int = TEMPORAL_WINDOW,
    recent_lag: int = RECENT_LAG,
) -> pd.DataFrame:
    """Append trailing features without crossing units or using future cycles."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if recent_lag < 1:
        raise ValueError("recent_lag must be at least 1.")
    required = {"unit_id", "cycle", *temporal_source_columns()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns required for temporal features: {missing}")
    if frame.index.has_duplicates:
        raise ValueError("Temporal feature construction requires a unique row index.")

    pieces = [
        _unit_temporal_features(unit_frame, temporal_source_columns(), window_size, recent_lag)
        for _, unit_frame in frame.groupby("unit_id", sort=True)
    ]
    history = pd.concat(pieces).reindex(frame.index)
    if history.isna().any().any() or not np.isfinite(history.to_numpy()).all():
        raise RuntimeError("Temporal feature construction produced missing or non-finite values.")
    return pd.concat([frame.copy(), history], axis=1)

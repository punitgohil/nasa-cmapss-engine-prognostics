from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate_temporal_features import (
    engine_balanced_weights,
    sample_validation_endpoints,
)
from src.prepare_data import SENSOR_COLUMNS
from src.temporal_features import add_temporal_features


def temporal_test_frame(units: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for unit_id, values in units.items():
        for cycle, value in enumerate(values, start=1):
            row = {"unit_id": unit_id, "cycle": cycle}
            row.update({f"{sensor}_deviation": value for sensor in SENSOR_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


class TemporalFeatureTests(unittest.TestCase):
    def test_trailing_features_use_only_current_and_past_rows(self) -> None:
        frame = temporal_test_frame({1: [1.0, 2.0, 3.0, 100.0]})
        result = add_temporal_features(frame, window_size=3, recent_lag=2)
        source = "sensor_1_deviation"
        self.assertAlmostEqual(result.loc[2, f"{source}_history_mean"], 2.0)
        self.assertAlmostEqual(result.loc[2, f"{source}_history_slope"], 1.0)
        self.assertAlmostEqual(result.loc[2, f"{source}_history_window_change"], 2.0)
        self.assertAlmostEqual(result.loc[2, f"{source}_history_recent_change"], 2.0)

    def test_windows_never_cross_engine_boundaries(self) -> None:
        frame = temporal_test_frame({1: [1.0, 2.0], 2: [50.0, 51.0]})
        result = add_temporal_features(frame, window_size=30)
        source = "sensor_1_deviation"
        first_unit_two = result[result["unit_id"] == 2].iloc[0]
        self.assertEqual(first_unit_two["history_observations"], 1.0)
        self.assertEqual(first_unit_two[f"{source}_history_mean"], 50.0)
        self.assertEqual(first_unit_two[f"{source}_history_window_change"], 0.0)

    def test_engine_balanced_weights_have_equal_unit_totals(self) -> None:
        frame = pd.DataFrame({"unit_id": [1, 1, 2, 2, 2, 2]})
        weights = engine_balanced_weights(frame)
        totals = pd.Series(weights).groupby(frame["unit_id"]).sum().to_numpy()
        self.assertTrue(np.allclose(totals, totals[0]))
        self.assertAlmostEqual(weights.mean(), 1.0)

    def test_validation_endpoint_sampling_is_balanced_and_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1] * 6 + [2] * 6,
                "cycle": list(range(1, 7)) * 2,
            }
        )
        first = sample_validation_endpoints(frame, endpoints_per_engine=3, random_seed=7)
        second = sample_validation_endpoints(frame, endpoints_per_engine=3, random_seed=7)
        self.assertEqual(first.groupby("unit_id").size().to_dict(), {1: 3, 2: 3})
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()

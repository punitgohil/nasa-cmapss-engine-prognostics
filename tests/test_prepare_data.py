from pathlib import Path
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prepare_data import (
    ConditionAwareTransformer,
    add_training_rul,
    split_development_and_calibration,
)
from src.evaluate_models import (
    conformal_adjustment,
    enforce_interval_nesting,
    sample_calibration_endpoints,
)


class PreparationTests(unittest.TestCase):
    def test_rul_is_calculated_from_each_unit_final_cycle(self) -> None:
        frame = pd.DataFrame({"unit_id": [1, 1, 2, 2], "cycle": [1, 3, 1, 2]})
        result = add_training_rul(frame)
        self.assertEqual(result["rul"].tolist(), [2.0, 0.0, 1.0, 0.0])

    def test_unit_split_has_no_overlap(self) -> None:
        frame = pd.DataFrame(
            {"unit_id": [unit for unit in range(1, 11) for _ in range(2)], "cycle": [1, 2] * 10}
        )
        development, calibration = split_development_and_calibration(frame, calibration_fraction=0.2)
        self.assertFalse(set(development["unit_id"]) & set(calibration["unit_id"]))
        self.assertEqual(len(development) + len(calibration), len(frame))

    def test_condition_transformer_requires_fit(self) -> None:
        transformer = ConditionAwareTransformer()
        with self.assertRaises(RuntimeError):
            transformer.transform(pd.DataFrame())

    def test_conformal_adjustment_expands_an_undercovering_interval(self) -> None:
        adjustment = conformal_adjustment(
            np.array([0.0, 10.0, 20.0]), np.array([1.0, 11.0, 21.0]), np.array([2.0, 12.0, 22.0])
        )
        self.assertGreaterEqual(adjustment, 1.0)

    def test_prediction_intervals_are_nested_by_confidence(self) -> None:
        raw = pd.DataFrame(
            {
                "predicted_rul": [10.0],
                "lower_80": [7.0], "upper_80": [13.0],
                "lower_90": [8.0], "upper_90": [12.0],
                "lower_95": [9.0], "upper_95": [11.0],
            }
        )
        nested = enforce_interval_nesting(raw)
        self.assertLessEqual(nested.loc[0, "lower_95"], nested.loc[0, "lower_90"])
        self.assertLessEqual(nested.loc[0, "lower_90"], nested.loc[0, "lower_80"])
        self.assertGreaterEqual(nested.loc[0, "upper_95"], nested.loc[0, "upper_90"])
        self.assertGreaterEqual(nested.loc[0, "upper_90"], nested.loc[0, "upper_80"])

    def test_calibration_sampling_selects_one_row_per_unit_deterministically(self) -> None:
        calibration = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "cycle": [1, 2, 3, 1, 2, 3],
                "rul": [2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
            }
        )
        first = sample_calibration_endpoints(calibration, random_seed=42)
        second = sample_calibration_endpoints(calibration, random_seed=42)
        self.assertEqual(len(first), 2)
        self.assertEqual(first["unit_id"].nunique(), 2)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()

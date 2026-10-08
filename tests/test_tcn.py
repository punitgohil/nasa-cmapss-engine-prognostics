from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tcn_model import CausalConv1d, TemporalCNN, TrajectorySequenceDataset


class TemporalCNNTests(unittest.TestCase):
    def test_locked_receptive_field_is_31_cycles(self) -> None:
        self.assertEqual(TemporalCNN(input_channels=3).receptive_field, 31)

    def test_causal_convolution_does_not_use_future_values(self) -> None:
        layer = CausalConv1d(1, 1, kernel_size=3, dilation=2)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.zero_()
        original = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        changed = original.clone()
        changed[..., 6:] = 1000.0
        original_output = layer(original)
        changed_output = layer(changed)
        self.assertTrue(torch.equal(original_output[..., :6], changed_output[..., :6]))

    def test_sequence_padding_and_mask_do_not_cross_units(self) -> None:
        frame = pd.DataFrame(
            {
                "unit_id": [1, 1, 2],
                "cycle": [1, 2, 1],
                "feature": [10.0, 20.0, 100.0],
                "rul": [1.0, 0.0, 0.0],
            }
        )
        scaler = StandardScaler().fit(frame[["feature"]])
        dataset = TrajectorySequenceDataset(
            frame,
            ["feature"],
            scaler,
            "rul",
            target_mean=0.0,
            target_scale=1.0,
            endpoint_indices=[2],
            sequence_length=4,
        )
        sequence, _, _ = dataset[0]
        self.assertEqual(tuple(sequence.shape), (2, 4))
        self.assertTrue(np.array_equal(sequence[-1].numpy(), [0.0, 0.0, 0.0, 1.0]))
        expected = scaler.transform(pd.DataFrame({"feature": [100.0]}))[0, 0]
        self.assertAlmostEqual(float(sequence[0, -1]), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()

"""Small causal Temporal CNN and leakage-safe trajectory sequence dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset


SEQUENCE_LENGTH = 30
HIDDEN_CHANNELS = 32
DILATIONS = (1, 2, 4, 8)
KERNEL_SIZE = 3
DROPOUT = 0.10


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution that cannot access later time positions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        self.left_padding = dilation * (kernel_size - 1)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        return output[..., : inputs.shape[-1]]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.convolution = CausalConv1d(
            in_channels, out_channels, kernel_size=KERNEL_SIZE, dilation=dilation
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.dropout(self.activation(self.convolution(inputs)))
        return self.activation(transformed + self.residual(inputs))


class TemporalCNN(nn.Module):
    """Locked Phase 2 TCN with a 31-cycle receptive field."""

    def __init__(self, input_channels: int):
        super().__init__()
        blocks = []
        channels = input_channels
        for dilation in DILATIONS:
            blocks.append(TemporalBlock(channels, HIDDEN_CHANNELS, dilation))
            channels = HIDDEN_CHANNELS
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_CHANNELS, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    @property
    def receptive_field(self) -> int:
        return 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        representation = self.network(inputs)[..., -1]
        return self.head(representation).squeeze(-1)


@dataclass(frozen=True)
class SequenceRecord:
    row_index: int
    unit_id: int
    cycle: int
    true_rul: float
    sample_weight: float


class TrajectorySequenceDataset(Dataset):
    """Precompute left-padded trailing sequences without crossing engines."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        feature_scaler: StandardScaler,
        target_column: str,
        target_mean: float,
        target_scale: float,
        endpoint_indices: Optional[Iterable[int]] = None,
        sample_weights: Optional[pd.Series] = None,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")
        if target_scale <= 0:
            raise ValueError("target_scale must be positive.")
        if frame.index.has_duplicates:
            raise ValueError("Sequence construction requires a unique row index.")
        required = {"unit_id", "cycle", target_column, *feature_columns}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing sequence columns: {missing}")

        selected = None if endpoint_indices is None else {int(index) for index in endpoint_indices}
        weight_lookup = sample_weights if sample_weights is not None else pd.Series(1.0, index=frame.index)
        records: list[SequenceRecord] = []
        sequence_arrays: list[np.ndarray] = []

        for _, unit_frame in frame.groupby("unit_id", sort=True):
            ordered = unit_frame.sort_values("cycle")
            scaled = feature_scaler.transform(ordered[feature_columns]).astype(np.float32)
            ordered_indices = ordered.index.to_numpy()
            for position, row_index in enumerate(ordered_indices):
                row_index = int(row_index)
                if selected is not None and row_index not in selected:
                    continue
                start = max(0, position - sequence_length + 1)
                observed = scaled[start : position + 1]
                sequence = np.zeros(
                    (len(feature_columns) + 1, sequence_length), dtype=np.float32
                )
                sequence[:-1, -len(observed) :] = observed.T
                sequence[-1, -len(observed) :] = 1.0
                row = ordered.loc[row_index]
                records.append(
                    SequenceRecord(
                        row_index=row_index,
                        unit_id=int(row["unit_id"]),
                        cycle=int(row["cycle"]),
                        true_rul=float(row[target_column]),
                        sample_weight=float(weight_lookup.loc[row_index]),
                    )
                )
                sequence_arrays.append(sequence)

        if selected is not None and len(records) != len(selected):
            found = {record.row_index for record in records}
            missing_indices = sorted(selected - found)
            raise ValueError(f"Some requested endpoint indices were not found: {missing_indices[:10]}")
        self.records = records
        self.sequences = np.stack(sequence_arrays)
        self.targets = np.asarray(
            [(record.true_rul - target_mean) / target_scale for record in records],
            dtype=np.float32,
        )
        self.weights = np.asarray([record.sample_weight for record in records], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sequences[index]),
            torch.tensor(self.targets[index], dtype=torch.float32),
            torch.tensor(self.weights[index], dtype=torch.float32),
        )

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_index": [record.row_index for record in self.records],
                "unit_id": [record.unit_id for record in self.records],
                "cycle": [record.cycle for record in self.records],
                "true_rul": [record.true_rul for record in self.records],
            }
        )

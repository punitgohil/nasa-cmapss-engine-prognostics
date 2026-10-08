"""Train the locked Phase 2 Temporal CNN and compare it with Method 3."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.audit_data import assert_clean_report, build_report
from src.evaluate_models import RUL_BIN_EDGES, RUL_BIN_LABELS
from src.evaluate_temporal_features import (
    PHASE2_SEED,
    engine_balanced_weights,
    sample_validation_endpoints,
    split_phase2_training_validation,
)
from src.prepare_data import (
    ConditionAwareTransformer,
    add_training_rul,
    condition_aware_feature_columns,
    load_test_rul,
    load_trajectory,
    split_development_and_calibration,
)
from src.tcn_model import (
    DILATIONS,
    DROPOUT,
    HIDDEN_CHANNELS,
    KERNEL_SIZE,
    SEQUENCE_LENGTH,
    TemporalCNN,
    TrajectorySequenceDataset,
)


BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8


def set_reproducible_seed(seed: int = PHASE2_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_scalers(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[StandardScaler, float, float, pd.Series]:
    weights = pd.Series(engine_balanced_weights(frame), index=frame.index)
    scaler = StandardScaler().fit(frame[feature_columns], sample_weight=weights.to_numpy())
    targets = frame[target_column].to_numpy(dtype=float)
    target_mean = float(np.average(targets, weights=weights.to_numpy()))
    target_variance = float(
        np.average((targets - target_mean) ** 2, weights=weights.to_numpy())
    )
    target_scale = target_variance**0.5
    if target_scale <= 0:
        raise ValueError("Training targets have zero variance.")
    return scaler, target_mean, target_scale, weights


def make_loader(
    dataset: TrajectorySequenceDataset,
    shuffle: bool,
    seed: int = PHASE2_SEED,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def train_epoch(
    model: TemporalCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_weighted_loss = 0.0
    total_weight = 0.0
    for sequences, targets, weights in loader:
        optimizer.zero_grad()
        predictions = model(sequences)
        absolute_errors = torch.abs(predictions - targets)
        loss = (absolute_errors * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()
        total_weighted_loss += float((absolute_errors.detach() * weights).sum())
        total_weight += float(weights.sum())
    return total_weighted_loss / total_weight


def predict_dataset(
    model: TemporalCNN,
    dataset: TrajectorySequenceDataset,
    target_mean: float,
    target_scale: float,
) -> pd.DataFrame:
    model.eval()
    standardized_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for sequences, _, _ in make_loader(dataset, shuffle=False):
            standardized_predictions.append(model(sequences).numpy())
    predictions = np.concatenate(standardized_predictions) * target_scale + target_mean
    output = dataset.metadata_frame()
    output["predicted_rul"] = predictions
    output["error"] = output["predicted_rul"] - output["true_rul"]
    output["absolute_error"] = output["error"].abs()
    output["arm"] = "temporal_cnn"
    output["rul_band"] = pd.cut(
        output["true_rul"],
        bins=RUL_BIN_EDGES,
        labels=RUL_BIN_LABELS,
        include_lowest=True,
    )
    return output


def train_with_early_stopping(
    training_dataset: TrajectorySequenceDataset,
    validation_dataset: TrajectorySequenceDataset,
    target_mean: float,
    target_scale: float,
) -> tuple[TemporalCNN, int, list[dict[str, float | int]]]:
    set_reproducible_seed()
    model = TemporalCNN(input_channels=training_dataset.sequences.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    training_loader = make_loader(training_dataset, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        training_loss = train_epoch(model, training_loader, optimizer)
        validation_predictions = predict_dataset(
            model, validation_dataset, target_mean, target_scale
        )
        validation_mae = float(validation_predictions["absolute_error"].mean())
        history.append(
            {
                "epoch": epoch,
                "engine_balanced_scaled_training_mae": training_loss,
                "validation_mae_cycles": validation_mae,
            }
        )
        if validation_mae < best_validation_mae - 1e-6:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("TCN training did not produce a validation result.")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def train_fixed_epochs(
    training_dataset: TrajectorySequenceDataset,
    epochs: int,
) -> tuple[TemporalCNN, list[float]]:
    set_reproducible_seed()
    model = TemporalCNN(input_channels=training_dataset.sequences.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_loader(training_dataset, shuffle=True)
    losses = [train_epoch(model, loader, optimizer) for _ in range(epochs)]
    return model, losses


def metric_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "engines": int(frame["unit_id"].nunique()),
        "mae": float(mean_absolute_error(frame["true_rul"], frame["predicted_rul"])),
        "rmse": float(mean_squared_error(frame["true_rul"], frame["predicted_rul"]) ** 0.5),
        "mean_error": float(frame["error"].mean()),
    }


def paired_bootstrap(
    combined: pd.DataFrame,
    baseline_arm: str = "phase2_engineered_history",
    candidate_arm: str = "temporal_cnn",
    repetitions: int = 10000,
) -> dict[str, Any]:
    engine_errors = (
        combined.groupby(["unit_id", "arm"], as_index=False)["absolute_error"].mean()
        .pivot(index="unit_id", columns="arm", values="absolute_error")
    )
    differences = (engine_errors[baseline_arm] - engine_errors[candidate_arm]).to_numpy()
    rng = np.random.default_rng(PHASE2_SEED)
    samples = np.empty(repetitions)
    for index in range(repetitions):
        samples[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return {
        "engines": int(len(differences)),
        "mean_mae_improvement_over_method3": float(differences.mean()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_95_interval": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "fraction_of_engines_improved": float((differences > 0).mean()),
    }


def align_reference(
    reference_predictions: pd.DataFrame,
    tcn_predictions: pd.DataFrame,
    evaluation_split: str,
) -> pd.DataFrame:
    tcn_predictions = tcn_predictions.drop(columns=["row_index"], errors="ignore").copy()
    reference = reference_predictions[
        (reference_predictions["evaluation_split"] == evaluation_split)
        & (reference_predictions["arm"] == "phase2_engineered_history")
    ].copy()
    keys = ["unit_id", "cycle"]
    check = reference[keys + ["true_rul"]].merge(
        tcn_predictions[keys + ["true_rul"]],
        on=keys,
        how="outer",
        suffixes=("_reference", "_tcn"),
        validate="one_to_one",
        indicator=True,
    )
    if not (check["_merge"] == "both").all() or not np.allclose(
        check["true_rul_reference"], check["true_rul_tcn"]
    ):
        raise ValueError("TCN endpoints do not align exactly with Method 3 endpoints.")
    return pd.concat([reference, tcn_predictions], ignore_index=True)


def attach_test_labels(test_frame: pd.DataFrame, test_rul: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Attach official labels only to each test engine's final observed row."""
    labelled = test_frame.copy()
    labelled["true_rul"] = np.nan
    endpoints: list[int] = []
    label_lookup = test_rul.set_index("unit_id")["true_rul"]
    for unit_id, unit_frame in labelled.groupby("unit_id", sort=True):
        endpoint = int(unit_frame["cycle"].idxmax())
        labelled.loc[endpoint, "true_rul"] = float(label_lookup.loc[unit_id])
        endpoints.append(endpoint)
    return labelled, endpoints


def run_tcn(
    raw_directory: Path,
    method3_predictions_path: Path,
    method3_results_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    assert_clean_report(build_report(raw_directory))
    feature_columns = condition_aware_feature_columns()
    train = add_training_rul(load_trajectory(raw_directory / "train_FD002.txt"))
    test = load_trajectory(raw_directory / "test_FD002.txt")
    test_rul = load_test_rul(raw_directory / "RUL_FD002.txt", test)
    development, calibration = split_development_and_calibration(train)
    phase2_training, phase2_validation = split_phase2_training_validation(development)
    reference_predictions = pd.read_csv(method3_predictions_path)
    method3_results = json.loads(method3_results_path.read_text(encoding="utf-8"))

    validation_transformer = ConditionAwareTransformer().fit(phase2_training)
    validation_training = validation_transformer.transform(phase2_training)
    validation_pool = validation_transformer.transform(phase2_validation)
    validation_endpoints = sample_validation_endpoints(validation_pool)
    validation_indices = validation_endpoints.index.tolist()
    scaler, target_mean, target_scale, training_weights = fit_scalers(
        validation_training, feature_columns, "rul"
    )
    training_dataset = TrajectorySequenceDataset(
        validation_training,
        feature_columns,
        scaler,
        "rul",
        target_mean,
        target_scale,
        sample_weights=training_weights,
    )
    validation_dataset = TrajectorySequenceDataset(
        validation_pool,
        feature_columns,
        scaler,
        "rul",
        target_mean,
        target_scale,
        endpoint_indices=validation_indices,
    )
    validation_model, best_epoch, training_history = train_with_early_stopping(
        training_dataset, validation_dataset, target_mean, target_scale
    )
    validation_tcn = predict_dataset(
        validation_model, validation_dataset, target_mean, target_scale
    )
    validation_combined = align_reference(
        reference_predictions, validation_tcn, "development_validation"
    )

    del training_dataset, validation_dataset, validation_model

    final_transformer = ConditionAwareTransformer().fit(development)
    final_training = final_transformer.transform(development)
    full_test = final_transformer.transform(test)
    labelled_test, test_indices = attach_test_labels(full_test, test_rul)
    final_scaler, final_target_mean, final_target_scale, final_weights = fit_scalers(
        final_training, feature_columns, "rul"
    )
    final_training_dataset = TrajectorySequenceDataset(
        final_training,
        feature_columns,
        final_scaler,
        "rul",
        final_target_mean,
        final_target_scale,
        sample_weights=final_weights,
    )
    test_dataset = TrajectorySequenceDataset(
        labelled_test,
        feature_columns,
        final_scaler,
        "true_rul",
        final_target_mean,
        final_target_scale,
        endpoint_indices=test_indices,
    )
    final_model, final_losses = train_fixed_epochs(final_training_dataset, best_epoch)
    test_tcn = predict_dataset(final_model, test_dataset, final_target_mean, final_target_scale)
    test_combined = align_reference(
        reference_predictions, test_tcn, "official_test_exploratory"
    )

    report = {
        "status": (
            "Exploratory Phase 2 analysis. Architecture and training protocol were "
            "locked before TCN validation and official-test results were inspected."
        ),
        "architecture": {
            "sequence_length": SEQUENCE_LENGTH,
            "input_feature_channels": len(feature_columns),
            "observation_mask_channels": 1,
            "hidden_channels": HIDDEN_CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "dilations": list(DILATIONS),
            "receptive_field_cycles": TemporalCNN(len(feature_columns) + 1).receptive_field,
            "dropout": DROPOUT,
        },
        "training": {
            "loss": "engine-balanced absolute error on standardized uncapped RUL",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "selected_validation_epoch": best_epoch,
            "validation_training_history": training_history,
            "final_training_scaled_loss_by_epoch": final_losses,
        },
        "splits": {
            "method3_training_engines": int(phase2_training["unit_id"].nunique()),
            "method3_validation_engines": int(phase2_validation["unit_id"].nunique()),
            "phase1_development_engines_for_final_fit": int(development["unit_id"].nunique()),
            "phase1_calibration_engines_not_used": int(calibration["unit_id"].nunique()),
            "official_test_engines": int(test["unit_id"].nunique()),
        },
        "development_validation": {
            "phase2_snapshot_reference": method3_results["development_validation"]["arms"][
                "phase2_snapshot"
            ],
            "method3_engineered_history": metric_summary(
                validation_combined[
                    validation_combined["arm"] == "phase2_engineered_history"
                ]
            ),
            "temporal_cnn": metric_summary(
                validation_combined[validation_combined["arm"] == "temporal_cnn"]
            ),
            "paired_comparison_to_method3": paired_bootstrap(validation_combined),
        },
        "official_test_exploratory": {
            "phase2_snapshot_reference": method3_results["official_test_exploratory"][
                "arms"
            ]["phase2_snapshot"],
            "method3_engineered_history": metric_summary(
                test_combined[test_combined["arm"] == "phase2_engineered_history"]
            ),
            "temporal_cnn": metric_summary(
                test_combined[test_combined["arm"] == "temporal_cnn"]
            ),
            "paired_comparison_to_method3": paired_bootstrap(test_combined),
        },
    }
    predictions = pd.concat(
        [
            validation_combined.assign(evaluation_split="development_validation"),
            test_combined.assign(evaluation_split="official_test_exploratory"),
        ],
        ignore_index=True,
    )
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked Phase 2 Temporal CNN.")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--method3-predictions",
        type=Path,
        default=Path("data/processed/phase2_method3_predictions.csv"),
    )
    parser.add_argument(
        "--method3-results",
        type=Path,
        default=Path("data/processed/phase2_method3_results.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/phase2_tcn_results.json")
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/processed/phase2_tcn_predictions.csv"),
    )
    args = parser.parse_args()

    report, predictions = run_tcn(
        args.raw_directory, args.method3_predictions, args.method3_results
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions.to_csv(args.predictions_output, index=False)
    print(f"Wrote TCN results to {args.output}")
    print(f"Wrote TCN predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()

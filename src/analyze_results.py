"""Diagnose where condition-aware FD002 modelling helps and where it does not."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.prepare_data import (
    ConditionAwareTransformer,
    OPERATION_SETTING_COLUMNS,
    add_training_rul,
    final_test_observations,
    load_test_rul,
    load_trajectory,
    split_development_and_calibration,
)


RANDOM_SEED = 42
ARM_LABELS = {
    "global_baseline": "Global baseline",
    "condition_aware": "Condition-aware",
}
ARM_COLORS = {
    "global_baseline": "#64748b",
    "condition_aware": "#2563eb",
}


def _mae(frame: pd.DataFrame) -> float:
    return float(frame["absolute_error"].mean())


def _group_comparison(predictions: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, group_frame in predictions.groupby(group_column, observed=False):
        if len(group_frame) == 0:
            continue
        global_frame = group_frame[group_frame["arm"] == "global_baseline"]
        condition_frame = group_frame[group_frame["arm"] == "condition_aware"]
        global_mae = _mae(global_frame)
        condition_mae = _mae(condition_frame)
        rows.append(
            {
                "group": str(group),
                "test_engines": int(len(condition_frame)),
                "global_mae": global_mae,
                "condition_aware_mae": condition_mae,
                "mae_improvement": global_mae - condition_mae,
            }
        )
    return rows


def _paired_bootstrap(predictions: pd.DataFrame, repetitions: int = 10000) -> dict[str, Any]:
    """Quantify uncertainty in the paired per-engine MAE improvement."""
    paired = predictions.pivot(index="unit_id", columns="arm", values="absolute_error")
    differences = (paired["global_baseline"] - paired["condition_aware"]).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_means = np.empty(repetitions)
    for index in range(repetitions):
        bootstrap_means[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    return {
        "test_engines": int(len(differences)),
        "mean_mae_improvement": float(differences.mean()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_95_interval": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "fraction_of_engines_with_lower_condition_aware_absolute_error": float(
            (differences > 0).mean()
        ),
    }


def _regime_structure(raw_directory: Path, predictions: pd.DataFrame) -> dict[str, Any]:
    train = add_training_rul(load_trajectory(raw_directory / "train_FD002.txt"))
    development, calibration = split_development_and_calibration(train)
    transformer = ConditionAwareTransformer().fit(development)

    scaled = transformer.setting_scaler_.transform(development[OPERATION_SETTING_COLUMNS])
    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = min(5000, len(scaled))
    sample_indexes = rng.choice(len(scaled), size=sample_size, replace=False)
    sampled_settings = scaled[sample_indexes]

    candidate_scores: list[dict[str, float | int]] = []
    for regime_count in range(2, 11):
        model = (
            transformer.regime_model_
            if regime_count == transformer.regime_count
            else KMeans(n_clusters=regime_count, random_state=RANDOM_SEED, n_init=20).fit(scaled)
        )
        labels = model.predict(sampled_settings)
        candidate_scores.append(
            {
                "regime_count": regime_count,
                "silhouette_score": float(silhouette_score(sampled_settings, labels)),
            }
        )

    development_regimes = transformer._predict_regimes(development)
    calibration_regimes = transformer._predict_regimes(calibration)
    endpoint_rows = predictions[predictions["arm"] == "condition_aware"]
    centers = transformer.setting_scaler_.inverse_transform(transformer.regime_model_.cluster_centers_)

    regimes: list[dict[str, Any]] = []
    for regime in range(transformer.regime_count):
        regimes.append(
            {
                "regime": regime,
                "development_rows": int((development_regimes == regime).sum()),
                "calibration_rows": int((calibration_regimes == regime).sum()),
                "test_endpoints": int((endpoint_rows["regime"] == regime).sum()),
                "anonymous_setting_center": {
                    column: float(value)
                    for column, value in zip(OPERATION_SETTING_COLUMNS, centers[regime])
                },
            }
        )
    return {
        "silhouette_sample_rows": sample_size,
        "candidate_regime_counts": candidate_scores,
        "six_regime_silhouette_score": next(
            row["silhouette_score"] for row in candidate_scores if row["regime_count"] == 6
        ),
        "regimes": regimes,
    }


def _save_predicted_vs_actual(predictions: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    limits = [predictions["true_rul"].min() - 5, predictions["true_rul"].max() + 5]
    for axis, arm in zip(axes, ARM_LABELS):
        frame = predictions[predictions["arm"] == arm]
        axis.scatter(frame["true_rul"], frame["predicted_rul"], s=24, alpha=0.65, color=ARM_COLORS[arm])
        axis.plot(limits, limits, color="#111827", linewidth=1, linestyle="--")
        axis.set_title(ARM_LABELS[arm])
        axis.set_xlabel("Actual RUL (cycles)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted RUL (cycles)")
    fig.suptitle("Predicted versus actual RUL")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_grouped_mae(comparison: list[dict[str, Any]], output: Path, title: str, x_label: str) -> None:
    import matplotlib.pyplot as plt

    groups = [row["group"] for row in comparison]
    positions = np.arange(len(groups))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - width / 2, [row["global_mae"] for row in comparison], width, label="Global", color=ARM_COLORS["global_baseline"])
    axis.bar(positions + width / 2, [row["condition_aware_mae"] for row in comparison], width, label="Condition-aware", color=ARM_COLORS["condition_aware"])
    axis.set_xticks(positions, groups)
    axis.set_xlabel(x_label)
    axis.set_ylabel("MAE (cycles)")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_interval_tradeoff(results: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for arm_report in results["arms"]:
        arm = arm_report["arm"]
        labels = list(arm_report["intervals"])
        nominal = [arm_report["intervals"][label]["nominal_coverage"] * 100 for label in labels]
        empirical = [arm_report["intervals"][label]["empirical_coverage"] * 100 for label in labels]
        widths = [arm_report["intervals"][label]["mean_interval_width"] for label in labels]
        axes[0].plot(nominal, empirical, marker="o", label=ARM_LABELS[arm], color=ARM_COLORS[arm])
        axes[1].plot(nominal, widths, marker="o", label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    axes[0].plot([78, 97], [78, 97], color="#111827", linestyle="--", linewidth=1, label="Ideal")
    axes[0].set_ylabel("Actual coverage (%)")
    axes[1].set_ylabel("Mean interval width (cycles)")
    for axis in axes:
        axis.set_xlabel("Desired coverage (%)")
        axis.set_xticks([80, 90, 95])
        axis.grid(alpha=0.2)
    axes[0].set_title("Empirical coverage")
    axes[1].set_title("Interval-level–width trade-off")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_individual_intervals(predictions: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    frame = predictions[predictions["arm"] == "condition_aware"].sort_values("true_rul")
    indexes = np.linspace(0, len(frame) - 1, 25, dtype=int)
    sample = frame.iloc[indexes].reset_index(drop=True)
    positions = np.arange(len(sample))
    lower_error = sample["predicted_rul"] - sample["lower_90"]
    upper_error = sample["upper_90"] - sample["predicted_rul"]
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.errorbar(
        positions,
        sample["predicted_rul"],
        yerr=[lower_error, upper_error],
        fmt="o",
        color=ARM_COLORS["condition_aware"],
        ecolor="#93c5fd",
        capsize=2,
        label="Prediction with 90% interval",
    )
    axis.scatter(positions, sample["true_rul"], marker="x", color="#111827", label="Actual RUL", zorder=3)
    axis.set_xticks(positions, sample["unit_id"].astype(int), rotation=90)
    axis.set_xlabel("Engine ID (25 representative test engines)")
    axis.set_ylabel("RUL (cycles)")
    axis.set_title("Individual condition-aware predictions and uncertainty")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_analysis(
    raw_directory: Path,
    predictions_path: Path,
    results_path: Path,
    figures_directory: Optional[Path],
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path)
    predictions["rul_band"] = pd.Categorical(
        predictions["rul_band"], categories=["<=25", "26-50", "51-100", ">100"], ordered=True
    )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    by_rul_band = _group_comparison(predictions, "rul_band")
    by_regime = _group_comparison(predictions, "regime")
    regime_structure = _regime_structure(raw_directory, predictions)
    if figures_directory is not None:
        figures_directory.mkdir(parents=True, exist_ok=True)
        _save_predicted_vs_actual(predictions, figures_directory / "predicted-vs-actual.png")
        _save_grouped_mae(by_rul_band, figures_directory / "error-by-rul-range.png", "Error by true-RUL range", "True-RUL range (cycles)")
        _save_grouped_mae(by_regime, figures_directory / "performance-by-regime.png", "Performance across inferred operating regimes", "Inferred regime")
        _save_interval_tradeoff(results, figures_directory / "interval-tradeoff.png")
        _save_individual_intervals(predictions, figures_directory / "individual-engine-intervals.png")

    return {
        "paired_bootstrap": _paired_bootstrap(predictions),
        "by_true_rul_band": by_rul_band,
        "by_regime": by_regime,
        "regime_structure": regime_structure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse the FD002 comparison and create figures.")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/engine_predictions.csv"))
    parser.add_argument("--results", type=Path, default=Path("data/processed/evaluation_results.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/analysis_results.json"))
    parser.add_argument("--figures-directory", type=Path, default=Path("figures"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    figures_directory = None if args.skip_figures else args.figures_directory
    analysis = build_analysis(args.raw_directory, args.predictions, args.results, figures_directory)
    args.output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {args.output}")
    if figures_directory is not None:
        print(f"Wrote figures to {args.figures_directory}")


if __name__ == "__main__":
    main()

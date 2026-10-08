"""Build the small local data bundle consumed by the static dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local dashboard data bundle.")
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/engine_predictions.csv"))
    parser.add_argument("--results", type=Path, default=Path("data/processed/evaluation_results.json"))
    parser.add_argument("--analysis", type=Path, default=Path("data/processed/analysis_results.json"))
    parser.add_argument("--output", type=Path, default=Path("dashboard/data.js"))
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions).round(4)
    payload = {
        "results": json.loads(args.results.read_text(encoding="utf-8")),
        "analysis": json.loads(args.analysis.read_text(encoding="utf-8")),
        "predictions": predictions.to_dict(orient="records"),
    }
    processed = args.results.parent
    payload['extensions'] = {
        name: json.loads((processed / filename).read_text(encoding='utf-8'))
        for name, filename in {
            'method3': 'phase2_method3_results.json',
            'tcn': 'phase2_tcn_results.json',
            'fd004': 'fd004_results.json',
        }.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote dashboard data to {args.output}")


if __name__ == "__main__":
    main()

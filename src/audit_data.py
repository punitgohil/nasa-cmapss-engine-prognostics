"""Audit the local NASA C-MAPSS FD002 source files.

This module performs structural checks only. It does not infer physical sensor
meanings, generate faults, or alter the raw data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COLUMN_NAMES = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{number}" for number in range(1, 22)]

EXPECTED_FIELD_COUNT = len(COLUMN_NAMES)


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_stats() -> dict[str, dict[str, float | int | None]]:
    return {
        name: {"count": 0, "minimum": None, "maximum": None, "mean": 0.0, "m2": 0.0}
        for name in COLUMN_NAMES
    }


def _finalise_stats(raw_stats: dict[str, dict[str, float | int | None]]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for name, stats in raw_stats.items():
        count = int(stats["count"])
        if count == 0:
            result[name] = {"count": 0, "minimum": None, "maximum": None, "variance": None}
            continue

        # Welford's online algorithm is stable even when values are large and
        # their variance is tiny, which is important for constant sensors.
        variance = float(stats["m2"]) / count
        result[name] = {
            "count": count,
            "minimum": stats["minimum"],
            "maximum": stats["maximum"],
            "variance": variance,
        }
    return result


def audit_trajectory_file(path: Path) -> dict[str, Any]:
    """Audit a C-MAPSS trajectory file without modifying it."""
    field_count_issues: list[dict[str, Any]] = []
    parse_issues: list[dict[str, Any]] = []
    duplicate_unit_cycles: list[dict[str, int]] = []
    cycle_continuity_issues: list[dict[str, int]] = []
    seen_unit_cycles: set[tuple[int, int]] = set()
    previous_cycle_by_unit: dict[int, int] = {}
    exact_operating_setting_tuples: set[tuple[float, float, float]] = set()
    raw_stats = _empty_stats()
    row_count = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                field_count_issues.append({"line": line_number, "observed_fields": 0})
                continue

            parts = stripped.split()
            row_count += 1
            if len(parts) != EXPECTED_FIELD_COUNT:
                field_count_issues.append({"line": line_number, "observed_fields": len(parts)})
                continue

            try:
                values = [float(value) for value in parts]
            except ValueError:
                parse_issues.append({"line": line_number})
                continue

            if not all(math.isfinite(value) for value in values):
                parse_issues.append({"line": line_number, "reason": "non_finite_value"})
                continue

            unit_id = int(values[0])
            cycle = int(values[1])
            if values[0] != unit_id or values[1] != cycle:
                parse_issues.append({"line": line_number, "reason": "non_integer_identifier_or_cycle"})
                continue

            pair = (unit_id, cycle)
            if pair in seen_unit_cycles:
                duplicate_unit_cycles.append({"unit_id": unit_id, "cycle": cycle})
            seen_unit_cycles.add(pair)

            if unit_id not in previous_cycle_by_unit:
                if cycle != 1:
                    cycle_continuity_issues.append(
                        {"unit_id": unit_id, "expected_cycle": 1, "observed_cycle": cycle}
                    )
            elif cycle != previous_cycle_by_unit[unit_id] + 1:
                cycle_continuity_issues.append(
                    {
                        "unit_id": unit_id,
                        "expected_cycle": previous_cycle_by_unit[unit_id] + 1,
                        "observed_cycle": cycle,
                    }
                )
            previous_cycle_by_unit[unit_id] = cycle
            exact_operating_setting_tuples.add((values[2], values[3], values[4]))

            for name, value in zip(COLUMN_NAMES, values):
                stats = raw_stats[name]
                prior_count = int(stats["count"])
                next_count = prior_count + 1
                prior_mean = float(stats["mean"])
                delta = value - prior_mean
                next_mean = prior_mean + delta / next_count
                stats["count"] = next_count
                stats["mean"] = next_mean
                stats["m2"] = float(stats["m2"]) + delta * (value - next_mean)
                stats["minimum"] = value if stats["minimum"] is None else min(float(stats["minimum"]), value)
                stats["maximum"] = value if stats["maximum"] is None else max(float(stats["maximum"]), value)

    statistics = _finalise_stats(raw_stats)
    zero_variance_columns = [
        name for name, stats in statistics.items() if stats["variance"] is not None and stats["variance"] == 0
    ]

    return {
        "file": path.name,
        "sha256": sha256(path),
        "rows": row_count,
        "distinct_units": len(previous_cycle_by_unit),
        "expected_fields_per_row": EXPECTED_FIELD_COUNT,
        "field_count_issue_count": len(field_count_issues),
        "field_count_issues": field_count_issues[:20],
        "numeric_parse_issue_count": len(parse_issues),
        "numeric_parse_issues": parse_issues[:20],
        "duplicate_unit_cycle_count": len(duplicate_unit_cycles),
        "duplicate_unit_cycles": duplicate_unit_cycles[:20],
        "cycle_continuity_issue_count": len(cycle_continuity_issues),
        "cycle_continuity_issues": cycle_continuity_issues[:20],
        "distinct_exact_operating_setting_tuple_count": len(exact_operating_setting_tuples),
        "operating_setting_note": (
            "Exact setting tuples are not condition labels. Condition-aware modeling "
            "must learn regimes from training operational-setting values and apply "
            "that learned rule to test data."
        ),
        "zero_variance_columns": zero_variance_columns,
        "column_statistics": statistics,
    }


def audit_rul_file(path: Path, expected_units: int) -> dict[str, Any]:
    """Audit the single-value-per-test-unit RUL file."""
    values: list[float] = []
    parse_issues: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                parse_issues.append({"line": line_number, "reason": "blank_value"})
                continue
            try:
                value = float(stripped)
            except ValueError:
                parse_issues.append({"line": line_number, "reason": "not_numeric"})
                continue
            if not math.isfinite(value):
                parse_issues.append({"line": line_number, "reason": "non_finite_value"})
                continue
            if value < 0 or not value.is_integer():
                parse_issues.append({"line": line_number, "reason": "not_a_non_negative_integer_cycle_count"})
                continue
            values.append(value)

    return {
        "file": path.name,
        "sha256": sha256(path),
        "values": len(values),
        "expected_values": expected_units,
        "count_matches_test_units": len(values) == expected_units,
        "numeric_parse_issue_count": len(parse_issues),
        "numeric_parse_issues": parse_issues[:20],
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def build_report(raw_directory: Path) -> dict[str, Any]:
    """Build an audit report for the three required FD002 files."""
    train_path = raw_directory / "train_FD002.txt"
    test_path = raw_directory / "test_FD002.txt"
    rul_path = raw_directory / "RUL_FD002.txt"
    missing_files = [path.name for path in (train_path, test_path, rul_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required input files are missing: {', '.join(missing_files)}")

    train = audit_trajectory_file(train_path)
    test = audit_trajectory_file(test_path)
    rul = audit_rul_file(rul_path, expected_units=int(test["distinct_units"]))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Structural audit only; raw data was not modified.",
        "dataset": "NASA C-MAPSS FD002",
        "train": train,
        "test": test,
        "rul": rul,
    }


def assert_clean_report(report: dict[str, Any]) -> None:
    """Raise if any structural issue required by the data contract is present."""
    failures: list[str] = []
    for split_name in ("train", "test"):
        split = report[split_name]
        for field in (
            "field_count_issue_count",
            "numeric_parse_issue_count",
            "duplicate_unit_cycle_count",
            "cycle_continuity_issue_count",
        ):
            if int(split[field]) != 0:
                failures.append(f"{split_name}.{field}={split[field]}")
    if int(report["rul"]["numeric_parse_issue_count"]) != 0:
        failures.append(f"rul.numeric_parse_issue_count={report['rul']['numeric_parse_issue_count']}")
    if not bool(report["rul"]["count_matches_test_units"]):
        failures.append("RUL value count does not match the number of test units")
    if failures:
        raise ValueError("FD002 structural audit failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local C-MAPSS FD002 source files.")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/data_quality_report.json"))
    args = parser.parse_args()

    report = build_report(args.raw_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote audit report to {args.output}")
    assert_clean_report(report)


if __name__ == "__main__":
    main()

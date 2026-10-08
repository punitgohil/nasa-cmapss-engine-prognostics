from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audit_data import assert_clean_report, audit_rul_file, audit_trajectory_file


class AuditDataTests(unittest.TestCase):
    def test_trajectory_audit_reports_clean_minimal_file(self) -> None:
        valid_row = " ".join(["1", "1"] + ["0.0"] * 24)
        second_row = " ".join(["1", "2"] + ["1.0"] * 24)
        # Keep temporary test files within the project. This avoids relying on a
        # sandboxed system-temp directory and keeps the test self-contained.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "trajectory.txt"
            path.write_text(f"{valid_row}\n{second_row}\n", encoding="utf-8")
            report = audit_trajectory_file(path)

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["distinct_units"], 1)
        self.assertEqual(report["field_count_issue_count"], 0)
        self.assertEqual(report["cycle_continuity_issue_count"], 0)

    def test_rul_audit_checks_alignment(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "rul.txt"
            path.write_text("10\n20\n", encoding="utf-8")
            report = audit_rul_file(path, expected_units=2)

        self.assertTrue(report["count_matches_test_units"])
        self.assertEqual(report["numeric_parse_issue_count"], 0)

    def test_rul_audit_rejects_negative_and_fractional_cycles(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "rul.txt"
            path.write_text("10\n-1\n2.5\n", encoding="utf-8")
            report = audit_rul_file(path, expected_units=3)
        self.assertEqual(report["numeric_parse_issue_count"], 2)


if __name__ == "__main__":
    unittest.main()

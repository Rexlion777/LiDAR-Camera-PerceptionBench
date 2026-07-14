import csv
import unittest
from pathlib import Path


class Selected1000AcceptanceTest(unittest.TestCase):
    def test_selected_1000_tables_exist(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance/selected_1000")
        files = {
            "matrix": root / "selected_1000_perturbation_matrix.csv",
            "ap": root / "selected_1000_per_setting_ap.csv",
            "health": root / "selected_1000_prediction_health.csv",
            "range": root / "selected_1000_failure_by_range.csv",
            "class": root / "selected_1000_failure_by_class.csv",
            "runtime": root / "selected_1000_runtime_health_metrics.csv",
        }
        for path in files.values():
            self.assertTrue(path.exists(), path.name)

        with files["matrix"].open(encoding="utf-8", newline="") as handle:
            matrix_rows = list(csv.DictReader(handle))
        with files["ap"].open(encoding="utf-8", newline="") as handle:
            ap_rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(matrix_rows), 20)
        self.assertGreaterEqual(len(ap_rows), 3)
        completed = [row for row in matrix_rows if row.get("status") == "completed"]
        partial = [row for row in matrix_rows if row.get("status") == "partial"]
        self.assertGreaterEqual(len(completed), 3)
        self.assertGreaterEqual(len(partial), 1)
        for row in completed:
            self.assertEqual(row.get("eval_scope"), "1000-frame-slice")
            self.assertEqual(row.get("sampling_mode"), "selected_1000")


if __name__ == "__main__":
    unittest.main()

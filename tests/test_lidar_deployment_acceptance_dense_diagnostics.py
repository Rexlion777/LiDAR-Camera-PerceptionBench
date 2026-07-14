import csv
import unittest
from pathlib import Path


class DeploymentAcceptanceDenseDiagnosticsTest(unittest.TestCase):
    def test_dense_diagnostics_exist_and_have_rows(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance/dense_diagnostics")
        expected = {
            "per_frame_prediction_dense.csv": 1000,
            "per_box_prediction_dense.csv": 1,
            "per_gt_object_detection_dense.csv": 1,
            "per_frame_latency_dense.csv": 100,
            "per_frame_diff_dense.csv": 50,
            "per_object_yaw_projection_shift_dense.csv": 100,
            "per_frame_time_offset_dense.csv": 50,
        }
        for name, minimum in expected.items():
            path = root / name
            self.assertTrue(path.exists(), name)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), minimum, name)

        with (root / "per_frame_prediction_dense.csv").open(encoding="utf-8", newline="") as handle:
            frame_rows = list(csv.DictReader(handle))
        self.assertIn("health_risk_frame", frame_rows[0])
        self.assertIn("point_count", frame_rows[0])

        with (root / "per_box_prediction_dense.csv").open(encoding="utf-8", newline="") as handle:
            box_rows = list(csv.DictReader(handle))
        self.assertIn("range_m", box_rows[0])
        self.assertIn("score", box_rows[0])

        with (root / "per_gt_object_detection_dense.csv").open(encoding="utf-8", newline="") as handle:
            gt_rows = list(csv.DictReader(handle))
        self.assertIn("matched", gt_rows[0])
        self.assertIn("range_bin", gt_rows[0])


if __name__ == "__main__":
    unittest.main()

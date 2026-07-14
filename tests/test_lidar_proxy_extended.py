import csv
import unittest
from pathlib import Path


class ProxyExtendedTest(unittest.TestCase):
    def test_proxy_extended_outputs_exist(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance/proxy_extended")
        files = {
            "yaw": root / "yaw_projection_shift.csv",
            "pitch": root / "pitch_projection_shift.csv",
            "roll": root / "roll_projection_shift.csv",
            "translation": root / "translation_projection_shift.csv",
            "time": root / "time_offset_proxy.csv",
        }
        for path in files.values():
            self.assertTrue(path.exists(), path.name)

        with files["yaw"].open(encoding="utf-8", newline="") as handle:
            yaw_rows = list(csv.DictReader(handle))
        with files["time"].open(encoding="utf-8", newline="") as handle:
            time_rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(yaw_rows), 100)
        self.assertGreaterEqual(len(time_rows), 10)
        self.assertIn("reprojection_shift_px", yaw_rows[0])
        self.assertIn("center_drift_m", time_rows[0])
        self.assertIn("temporal_consistency_error", time_rows[0])


if __name__ == "__main__":
    unittest.main()

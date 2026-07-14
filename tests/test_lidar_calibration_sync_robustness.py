import json
import unittest
from pathlib import Path


class CalibrationSyncRobustnessTest(unittest.TestCase):
    def test_robustness_report_schema_if_present(self):
        path = Path("reports/lidar_system_algorithm/calibration_sync_robustness.json")
        if not path.exists():
            self.skipTest("calibration robustness report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(payload.get("status"), {"completed", "skipped"})
        self.assertIn("yaw_summary", payload)
        self.assertIn("time_offset_summary", payload)
        self.assertTrue(payload.get("limitations"))


if __name__ == "__main__":
    unittest.main()

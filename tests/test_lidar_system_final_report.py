import json
import unittest
from pathlib import Path


class LidarSystemFinalReportTest(unittest.TestCase):
    def test_final_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/lidar_system_algorithm_final_report.json")
        if not path.exists():
            self.skipTest("final system report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("backbone_head_only_trt", payload)
        self.assertIn("online_latency", payload)
        self.assertIn("full_bucketed_trt_blocker", payload)
        self.assertIn("figures", payload)


if __name__ == "__main__":
    unittest.main()

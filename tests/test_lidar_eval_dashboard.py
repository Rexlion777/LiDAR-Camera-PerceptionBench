import json
import unittest
from pathlib import Path


class EvalDashboardTest(unittest.TestCase):
    def test_dashboard_schema_if_present(self):
        path = Path("reports/lidar_system_algorithm/eval_dashboard.json")
        if not path.exists():
            self.skipTest("eval dashboard report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("kitti_ap_summary", payload)
        self.assertIsNotNone(payload["kitti_ap_summary"].get("Car_3d/moderate_R40"))
        self.assertIsNotNone(payload["kitti_ap_summary"].get("Pedestrian_3d/moderate_R40"))
        self.assertIsNotNone(payload["kitti_ap_summary"].get("Cyclist_3d/moderate_R40"))
        self.assertIn("latency_summary", payload)
        self.assertIn("dashboard_figures", payload)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path


class EvalDashboardBackboneHeadTRTTest(unittest.TestCase):
    def test_dashboard_if_present(self):
        path = Path("reports/lidar_system_algorithm/eval_dashboard.json")
        if not path.exists():
            self.skipTest("eval dashboard not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("tensorrt_backbone_head_only_summary", payload)
        self.assertIn("full_core_blocker", payload["tensorrt_backbone_head_only_summary"])


if __name__ == "__main__":
    unittest.main()

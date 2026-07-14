import json
import unittest
from pathlib import Path


class TensorRTBackboneHeadOnlyReportTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_backbone_head_only_final_report.json")
        if not path.exists():
            self.skipTest("backbone/head-only final report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("ap_comparison", payload)
        self.assertIn("latency_comparison", payload)
        self.assertTrue(payload.get("full_bucketed_trt_blocker", {}).get("still_exists"))


if __name__ == "__main__":
    unittest.main()

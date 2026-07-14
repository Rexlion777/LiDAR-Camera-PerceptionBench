import json
import unittest
from pathlib import Path


class TensorRTBackboneHeadOnlineLatencyTest(unittest.TestCase):
    def test_online_latency_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_backbone_head_online_latency.json")
        if not path.exists():
            self.skipTest("backbone/head online latency report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertTrue(payload.get("visualization_excluded"))
        self.assertIsNotNone(payload.get("pytorch_summary", {}).get("online_total_ms", {}).get("p95"))
        self.assertIsNotNone(payload.get("trt_summary", {}).get("online_total_ms", {}).get("p95"))
        self.assertGreater(payload.get("speedup", {}).get("core_only", 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()

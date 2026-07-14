import json
import unittest
from pathlib import Path


class TensorRTBackboneHeadDiffSummaryTest(unittest.TestCase):
    def test_diff_summary_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_backbone_head_only_diff.json")
        if not path.exists():
            self.skipTest("backbone/head-only diff report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        self.assertEqual(payload.get("status"), "completed")
        self.assertEqual(summary.get("count_used_for_mean"), summary.get("count_used_for_p95"))
        self.assertIsNotNone(summary.get("p50_topk_center_diff"))
        self.assertIsNotNone(summary.get("p95_topk_center_diff"))
        self.assertIsNotNone(summary.get("p99_topk_center_diff"))
        self.assertGreaterEqual(summary.get("p99_topk_center_diff"), summary.get("p95_topk_center_diff"))
        self.assertGreaterEqual(summary.get("max_topk_center_diff"), summary.get("p99_topk_center_diff"))


if __name__ == "__main__":
    unittest.main()

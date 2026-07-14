import json
import unittest
from pathlib import Path

from runtime.lidar_system_algorithm.tensorrt_bucketed_wrapper import compute_truncation_stats, parse_bucket_sizes, select_bucket_size


class TensorRTBucketedWrapperTest(unittest.TestCase):
    def test_bucket_selection(self):
        buckets = parse_bucket_sizes("4096,8192,12000")
        self.assertEqual(select_bucket_size(3000, buckets), 4096)
        self.assertEqual(select_bucket_size(4096, buckets), 4096)
        self.assertEqual(select_bucket_size(5000, buckets), 8192)
        self.assertEqual(select_bucket_size(50000, buckets), 12000)

    def test_truncation_stats(self):
        stats = compute_truncation_stats(15000, 12000)
        self.assertTrue(stats.truncated)
        self.assertEqual(stats.truncated_pillars, 3000)
        self.assertEqual(stats.padding_pillars, 0)
        stats2 = compute_truncation_stats(8000, 12000)
        self.assertFalse(stats2.truncated)
        self.assertEqual(stats2.padding_pillars, 4000)

    def test_report_schema_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_bucketed_core_report.json")
        if not path.exists():
            self.skipTest("bucketed TensorRT report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(payload.get("status"), {"completed", "skipped", "failed"})
        if payload.get("status") == "completed":
            self.assertIn("limitations", payload)
            self.assertTrue("core" in payload.get("scope", "").lower())


if __name__ == "__main__":
    unittest.main()

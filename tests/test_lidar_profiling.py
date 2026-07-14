from __future__ import annotations

import unittest

from runtime.lidar_system_algorithm.profiling import StageProfiler, aggregate_stage_records, cuda_synchronize_if_needed


class TestLidarProfiling(unittest.TestCase):
    def test_profiler_outputs_summary_stats(self) -> None:
        profiler = StageProfiler()
        profiler.runs = [
            {"data_load_ms": 1.0, "total_ms": 3.0},
            {"data_load_ms": 2.0, "total_ms": 5.0},
            {"data_load_ms": 4.0, "total_ms": 7.0},
        ]
        rows = aggregate_stage_records(profiler.runs, ["data_load_ms", "total_ms"])
        self.assertEqual(rows[0]["count"], 3)
        self.assertAlmostEqual(rows[0]["mean_ms"], 7.0 / 3.0)
        self.assertIsNotNone(rows[0]["p50_ms"])
        self.assertIsNotNone(rows[0]["p95_ms"])

    def test_cuda_sync_no_crash_when_unavailable(self) -> None:
        cuda_synchronize_if_needed()


if __name__ == "__main__":
    unittest.main()

import unittest

from runtime.lidar_system_algorithm.online_latency import compute_online_debug_records, summarize_online_latency


class OnlineLatencySchemaTest(unittest.TestCase):
    def test_online_total_excludes_visualization(self):
        records = compute_online_debug_records(
            [
                {
                    "frame_id": "000001",
                    "data_load_ms": 1.0,
                    "calibration_parse_ms": 2.0,
                    "point_preprocess_ms": 3.0,
                    "voxelization_or_pillarization_ms": 4.0,
                    "model_forward_ms": 5.0,
                    "nms_ms": 6.0,
                    "postprocess_ms": 7.0,
                    "visualization_ms": 100.0,
                }
            ],
            tracking_by_frame={"000001": 8.0},
        )
        self.assertAlmostEqual(records[0]["online_total_ms"], 36.0)
        self.assertAlmostEqual(records[0]["debug_total_ms"], 100.0)
        payload = summarize_online_latency(records)
        self.assertEqual(payload["schema_version"], "online_vs_debug_v1")
        self.assertIn("online_total_ms", payload["online_stage_names"])


if __name__ == "__main__":
    unittest.main()

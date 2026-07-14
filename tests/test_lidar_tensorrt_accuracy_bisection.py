import unittest
from pathlib import Path

from runtime.lidar_system_algorithm.tensorrt_accuracy_debug import (
    analyze_decode_nms_rows,
    analyze_raw_tensor_rows,
    build_route_summaries,
)


class TensorRTAccuracyBisectionTest(unittest.TestCase):
    def test_raw_tensor_analysis_flags_severe_mismatch(self):
        rows, summary = analyze_raw_tensor_rows(
            [
                {
                    "frame_id": "000001",
                    "bucket_size": 8192,
                    "pillar_count": 5000,
                    "cls_abs_diff_mean": 40000.0,
                    "box_abs_diff_mean": float("inf"),
                    "py_cls_min": -10.0,
                    "py_cls_max": 2.0,
                    "trt_cls_min": -2.0e8,
                    "trt_cls_max": 1.0e7,
                }
            ]
        )
        self.assertEqual(summary["severe_frame_count"], 1)
        self.assertIn("TRT raw output", summary["suspected_layer"])
        self.assertEqual(rows[0]["severity"], "severe")

    def test_decode_analysis_flags_empty_trt(self):
        rows, summary = analyze_decode_nms_rows(
            [
                {
                    "frame_id": "000001",
                    "selected_bucket_size": 8192,
                    "full_pillar_count": 5000,
                    "padding_pillars": 3192,
                    "pytorch_box_count": 12,
                    "trt_box_count": 0,
                }
            ]
        )
        self.assertEqual(summary["trt_zero_box_frames"], 1)
        self.assertEqual(rows[0]["issue"], "trt_post_nms_empty")

    def test_route_summaries_use_report_inputs(self):
        report_dir = Path("reports/lidar_system_algorithm")
        if not report_dir.exists():
            self.skipTest("report dir not available")
        rows = build_route_summaries(
            baseline_eval={"status": "completed", "frame_count": 10, "official_result_dict": {"Car_3d/moderate_R40": 1.0}},
            bucket_report={"frame_rows": [{"pytorch_box_count": 5, "trt_box_count": 0}]},
            bucket_eval={"status": "completed_with_blocker", "frame_count": 10, "official_result_dict": {"Car_3d/moderate_R40": 0.0}, "blocker": "x"},
            raw_diff={"rows": [{"frame_id": "000001", "cls_abs_diff_mean": 10000.0, "box_abs_diff_mean": float("inf")}]},
            baseline_pred_dir=Path("projects/lidar_system_algorithm/results/kitti_eval_txt"),
            trt_pred_dir=Path("projects/lidar_system_algorithm/results/kitti_eval_txt_trt_bucketed"),
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["route"], "B_wrapper_pytorch_core")
        self.assertEqual(rows[2]["route"], "C_wrapper_trt_core")


if __name__ == "__main__":
    unittest.main()

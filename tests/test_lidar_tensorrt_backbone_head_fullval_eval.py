import json
import unittest
from pathlib import Path


class TensorRTBackboneHeadFullvalEvalTest(unittest.TestCase):
    def test_fullval_or_slice_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_backbone_head_only_fullval_eval.json")
        if not path.exists():
            self.skipTest("backbone/head-only fullval eval report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        trt = payload.get("trt_backbone_head_only", {})
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn(payload.get("result_scope"), {"full-val", "slice-1000"})
        self.assertEqual(trt.get("prediction_audit", {}).get("empty_prediction_file_count"), 0)
        self.assertEqual(trt.get("prediction_audit", {}).get("invalid_geometry_count"), 0)
        self.assertIsNotNone(trt.get("official_result_dict", {}).get("Car_3d/moderate_R40"))
        self.assertIsNotNone(trt.get("latency_summary", {}).get("trt_core_ms", {}).get("p95"))


if __name__ == "__main__":
    unittest.main()

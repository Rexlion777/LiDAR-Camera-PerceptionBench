import json
import unittest
from pathlib import Path


class TensorRTBackboneHeadOnlyEvalTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_backbone_head_only_eval.json")
        if not path.exists():
            self.skipTest("backbone/head-only TRT report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("official_result_dict", payload)
        self.assertEqual(payload.get("empty_prediction_file_count"), 0)


if __name__ == "__main__":
    unittest.main()

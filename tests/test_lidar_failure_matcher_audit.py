import tempfile
import unittest
from pathlib import Path

from runtime.lidar_system_algorithm.tensorrt_accuracy_debug import audit_failure_matcher


class FailureMatcherAuditTest(unittest.TestCase):
    def test_failure_matcher_audit_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            label_dir = root / "label_2"
            pred_dir = root / "pred"
            label_dir.mkdir()
            pred_dir.mkdir()
            (label_dir / "000001.txt").write_text(
                "Car 0 0 0 0 0 50 60 1.5 1.6 4.0 1.0 1.5 20.0 0.1\n"
                "DontCare -1 -1 -10 0 0 40 40 -1 -1 -1 -1000 -1000 -1000 -10\n",
                encoding="utf-8",
            )
            (pred_dir / "000001.txt").write_text(
                "Car 0 0 0 0 0 50 60 1.5 1.6 4.0 1.0 1.5 20.0 0.1 0.95\n"
                "Car 0 0 0 0 0 50 60 1.5 1.6 4.0 1.0 1.5 20.0 0.1 0.40\n",
                encoding="utf-8",
            )
            payload = audit_failure_matcher(label_dir, pred_dir, thresholds=[0.0, 0.5])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["unsupported_gt_filtering"]["dontcare_line_count"], 1)
            self.assertEqual(len(payload["tp_fp_fn_by_score_threshold"]), 2)
            self.assertTrue(payload["matcher_checks"]["class_aware_matching"])


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path


class DirClsYawFixTest(unittest.TestCase):
    def test_audit_and_parity_reports_if_present(self):
        audit_path = Path("reports/lidar_system_algorithm/wrapper_pytorch_core_batch_dict_audit.json")
        parity_path = Path("reports/lidar_system_algorithm/wrapper_pytorch_core_parity_report.json")
        if not audit_path.exists() or not parity_path.exists():
            self.skipTest("dir_cls/yaw supporting reports not generated")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        self.assertEqual(audit.get("status"), "completed")
        self.assertEqual(parity.get("status"), "completed")
        self.assertIn("summary", parity)
        self.assertLess(parity["summary"].get("topk_rotation_y_abs_diff_mean", 1.0), 0.1)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from runtime.lidar_system_algorithm.tensorrt_accuracy_debug import audit_prediction_export


class KittiExportAuditTest(unittest.TestCase):
    def test_export_audit_detects_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_dir = Path(tmp_dir)
            (pred_dir / "000001.txt").write_text("", encoding="utf-8")
            (pred_dir / "000002.txt").write_text(
                "Car 0 0 0 0 0 50 50 1.5 1.6 4.0 1.0 1.5 20.0 0.1 0.9\n",
                encoding="utf-8",
            )
            rows, summary = audit_prediction_export(pred_dir)
            self.assertEqual(summary["prediction_file_count"], 2)
            self.assertEqual(summary["empty_prediction_file_count"], 1)
            self.assertEqual(summary["direct_symptom"], "prediction files contain at least some boxes")
            self.assertEqual(rows[1]["line_fields_valid"], True)


if __name__ == "__main__":
    unittest.main()

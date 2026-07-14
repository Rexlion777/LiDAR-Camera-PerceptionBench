from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "diagnose_finetune_drift"


class TestFineTuneDriftPerClassDiagnostics(unittest.TestCase):
    def test_per_class_outputs_exist(self) -> None:
        for suffix in ("csv", "md", "json"):
            self.assertTrue((DIAG_ROOT / f"per_class_diagnostics.{suffix}").exists())

    def test_per_class_csv_has_expected_content(self) -> None:
        with (DIAG_ROOT / "per_class_diagnostics.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 24)
        required = {
            "model_name",
            "class_name",
            "boxes_per_frame_class",
            "score_mean_class",
            "prediction_gt_ratio_class",
            "empty_prediction_frames",
            "invalid_geometry_boxes",
        }
        self.assertTrue(required.issubset(rows[0].keys()))
        classes = {row["class_name"] for row in rows}
        self.assertEqual(classes, {"Car", "Pedestrian", "Cyclist"})

    def test_per_class_json_has_gt_counts(self) -> None:
        payload = json.loads((DIAG_ROOT / "per_class_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertIn("Car", payload["gt_counts"])
        self.assertIn("Pedestrian", payload["gt_counts"])
        self.assertIn("Cyclist", payload["gt_counts"])


if __name__ == "__main__":
    unittest.main()

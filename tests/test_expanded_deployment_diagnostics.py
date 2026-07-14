from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
FIGURE_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune"


class TestExpandedDeploymentDiagnostics(unittest.TestCase):
    def test_deployment_diagnostic_outputs_exist(self) -> None:
        for suffix in ("csv", "md", "json"):
            self.assertTrue((REPORT_ROOT / f"expanded_deployment_diagnostics.{suffix}").exists())
        for base_name in (
            "12_holdout_prediction_count_comparison",
            "13_score_distribution_pretrained_vs_finetuned",
            "14_range_distribution_pretrained_vs_finetuned",
        ):
            for ext in ("png", "svg", "pdf"):
                self.assertTrue((FIGURE_ROOT / f"{base_name}.{ext}").exists())

    def test_deployment_csv_has_expected_columns(self) -> None:
        with (REPORT_ROOT / "expanded_deployment_diagnostics.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        required_columns = {
            "model_name",
            "total_boxes",
            "mean_boxes_per_frame",
            "empty_prediction_frames",
            "invalid_geometry_boxes",
            "score_mean",
            "score_p50",
            "score_p95",
            "car_box_count",
            "ped_box_count",
            "cyc_box_count",
            "prediction_count_drift",
            "score_distribution_drift",
            "class_distribution_drift",
            "range_distribution_drift",
            "mean_ap_3d_moderate",
        }
        self.assertTrue(required_columns.issubset(rows[0].keys()))

    def test_deployment_json_preserves_false_positive_warning(self) -> None:
        payload = json.loads((REPORT_ROOT / "expanded_deployment_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["holdout_split"], "holdout_eval_500")
        self.assertEqual(payload["frame_count"], 500)
        self.assertIn("false positive risk", payload["analysis"]["box_count_interpretation"].lower())
        self.assertEqual(len(payload["rows"]), 3)


if __name__ == "__main__":
    unittest.main()

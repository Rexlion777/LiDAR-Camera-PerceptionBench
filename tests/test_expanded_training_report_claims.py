from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestExpandedTrainingReportClaims(unittest.TestCase):
    def test_report_files_exist(self) -> None:
        self.assertTrue((REPORT_ROOT / "expanded_training_finetune_report.md").exists())
        self.assertTrue((REPORT_ROOT / "expanded_training_finetune_report.json").exists())

    def test_report_json_has_boundary_fields(self) -> None:
        payload = json.loads((REPORT_ROOT / "expanded_training_finetune_report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["native_tools_test_eval_fixed"])
        self.assertFalse(payload["full_val"])
        self.assertIn("safe_claims", payload)
        self.assertIn("forbidden_claims", payload)
        forbidden_blob = " ".join(payload["forbidden_claims"]).lower()
        self.assertIn("full kitti training", forbidden_blob)
        self.assertIn("sota", forbidden_blob)
        self.assertIn("full kitti val", forbidden_blob)

    def test_report_text_marks_subset_and_holdout_boundary(self) -> None:
        report_text = (REPORT_ROOT / "expanded_training_finetune_report.md").read_text(encoding="utf-8", errors="replace").lower()
        self.assertIn("subset", report_text)
        self.assertIn("holdout", report_text)
        self.assertIn("full_val=false", report_text)
        self.assertNotIn("full kitti training completed", report_text)
        self.assertNotIn("achieved sota", report_text)

    def test_resume_bullets_updated_from_actual_result(self) -> None:
        resume_text = (PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "resume_bullets.md").read_text(
            encoding="utf-8",
            errors="replace",
        ).lower()
        self.assertIn("1000-sample subset fine-tuning", resume_text)
        self.assertIn("500-frame holdout", resume_text)
        self.assertNotIn("full kitti training", resume_text)


if __name__ == "__main__":
    unittest.main()

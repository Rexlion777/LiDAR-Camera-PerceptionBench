from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "diagnose_finetune_drift"


class TestFineTuneDriftReportClaims(unittest.TestCase):
    def test_report_outputs_exist(self) -> None:
        self.assertTrue((DIAG_ROOT / "finetune_drift_diagnosis_report.md").exists())
        self.assertTrue((DIAG_ROOT / "finetune_drift_diagnosis_report.json").exists())

    def test_report_boundaries_and_fp_risk(self) -> None:
        text = (DIAG_ROOT / "finetune_drift_diagnosis_report.md").read_text(encoding="utf-8", errors="replace").lower()
        self.assertIn("holdout_eval_500", text)
        self.assertIn("fp-risk", text)
        self.assertIn("false-positive risk", text)
        self.assertNotIn("achieved sota", text)
        self.assertNotIn("full-val completed", text)
        self.assertNotIn("full training completed", text)

    def test_report_json_has_safe_forbidden_claims(self) -> None:
        payload = json.loads((DIAG_ROOT / "finetune_drift_diagnosis_report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertFalse(payload["full_val"])
        self.assertIn("safe_claims", payload)
        self.assertIn("forbidden_claims", payload)

    def test_resume_does_not_overclaim(self) -> None:
        text = (PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "resume_bullets.md").read_text(encoding="utf-8", errors="replace")
        self.assertIn("false-positive", text)
        self.assertNotIn("性能显著提升", text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestPointPillarsTrainingReportClaims(unittest.TestCase):
    def test_report_has_scope_boundaries(self) -> None:
        report_path = REPORT_ROOT / "pointpillars_training_finetune_report.md"
        self.assertTrue(report_path.exists())
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("subset-val", text)
        self.assertIn("Do not claim SOTA.", text)
        self.assertIn("Do not claim KITTI full training.", text)
        self.assertIn("Do not call smoke train convergence.", text)
        self.assertNotIn("full KITTI training completed", text)
        self.assertNotIn("SOTA model", text)

    def test_resume_updated(self) -> None:
        resume_path = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "resume_bullets.md"
        self.assertTrue(resume_path.exists())
        text = resume_path.read_text(encoding="utf-8")
        self.assertIn("subset training / smoke fine-tuning", text)


if __name__ == "__main__":
    unittest.main()

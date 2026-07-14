from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


def latest_dir(prefix: str) -> Path:
    matches = sorted(REPORT_ROOT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise AssertionError(f"Missing experiment dir for {prefix}")
    return matches[-1]


class TestPointPillarsSubsetTrainingOutputs(unittest.TestCase):
    def test_training_outputs_exist(self) -> None:
        smoke_dir = latest_dir("smoke_train")
        finetune_dir = latest_dir("subset_finetune")
        skipped_dir = latest_dir("subset_train_from_scratch")

        smoke_status = json.loads((smoke_dir / "train_status.json").read_text(encoding="utf-8"))
        finetune_status = json.loads((finetune_dir / "train_status.json").read_text(encoding="utf-8"))
        skipped_status = json.loads((skipped_dir / "train_status.json").read_text(encoding="utf-8"))

        self.assertIn(smoke_status["status"], {"completed", "partial"})
        self.assertIn(finetune_status["status"], {"completed", "partial"})
        self.assertEqual(skipped_status["status"], "skipped")

        for exp_dir in (smoke_dir, finetune_dir):
            self.assertTrue((exp_dir / "train.log").exists())
            self.assertTrue((exp_dir / "metrics_summary.json").exists())
            self.assertTrue((exp_dir / "loss_curve.csv").exists())
            self.assertTrue((exp_dir / "eval_ap_summary.csv").exists())
            self.assertTrue((exp_dir / "safe_claims.md").exists())
            self.assertTrue((exp_dir / "forbidden_claims.md").exists())

        comparison_csv = REPORT_ROOT / "training_eval_comparison.csv"
        self.assertTrue(comparison_csv.exists())
        text = comparison_csv.read_text(encoding="utf-8")
        self.assertIn("subset_finetune", text)
        self.assertIn("smoke_train", text)


if __name__ == "__main__":
    unittest.main()

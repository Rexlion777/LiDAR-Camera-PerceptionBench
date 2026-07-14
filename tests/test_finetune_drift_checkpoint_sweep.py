from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "diagnose_finetune_drift"


class TestFineTuneDriftCheckpointSweep(unittest.TestCase):
    def test_checkpoint_sweep_outputs_exist(self) -> None:
        for suffix in ("csv", "md", "json"):
            self.assertTrue((DIAG_ROOT / f"checkpoint_sweep_holdout_ap.{suffix}").exists())

    def test_checkpoint_sweep_has_expected_models(self) -> None:
        with (DIAG_ROOT / "checkpoint_sweep_holdout_ap.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        names = {row["model_name"] for row in rows}
        for expected in (
            "pretrained_baseline",
            "subset_finetune_200_50_epoch3",
            "expanded_lr0008_epoch1",
            "expanded_lr0008_epoch2",
            "expanded_lr0008_epoch3",
            "expanded_lr0004_epoch1",
            "expanded_lr0004_epoch2",
            "expanded_lr0004_epoch3",
        ):
            self.assertIn(expected, names)
        self.assertTrue(any(row["best_mean_ap"] == "True" for row in rows))
        self.assertTrue(any(row["best_balanced_checkpoint"] == "True" for row in rows))

    def test_json_selection_present(self) -> None:
        payload = json.loads((DIAG_ROOT / "checkpoint_sweep_holdout_ap.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertIn("best_mean_ap_model", payload["selection"])
        self.assertIn("best_balanced_checkpoint", payload["selection"])


if __name__ == "__main__":
    unittest.main()

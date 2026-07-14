from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "diagnose_finetune_drift"
FIG_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune" / "diagnose_finetune_drift"


class TestFineTuneDriftLRSweep(unittest.TestCase):
    def test_lr_sweep_outputs_exist(self) -> None:
        self.assertTrue((DIAG_ROOT / "lr_sweep_summary.csv").exists())
        self.assertTrue((DIAG_ROOT / "lr_sweep_report.md").exists())
        self.assertTrue((DIAG_ROOT / "lr_sweep_report.json").exists())

    def test_lr_sweep_has_lr0004_and_lr0008(self) -> None:
        with (DIAG_ROOT / "lr_sweep_summary.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        lrs = {row["lr"] for row in rows}
        self.assertIn("0.0008", lrs)
        self.assertIn("0.0004", lrs)
        self.assertEqual(len(rows), 6)
        payload = json.loads((DIAG_ROOT / "lr_sweep_report.json").read_text(encoding="utf-8"))
        self.assertIn("skipped_lr0002_reason", payload)

    def test_figures_and_plot_data_exist(self) -> None:
        names = (
            "checkpoint_sweep_ap_by_class",
            "checkpoint_sweep_mean_ap",
            "per_class_box_count_comparison",
            "per_class_score_mean_comparison",
            "prediction_gt_ratio_by_class",
            "lr_sweep_ap_comparison",
            "drift_diagnosis_summary",
        )
        for name in names:
            for ext in ("png", "svg", "pdf"):
                self.assertTrue((FIG_ROOT / f"{name}.{ext}").exists())
        for name in names[:-1]:
            self.assertTrue((DIAG_ROOT / "plot_data" / f"{name}.csv").exists())
            self.assertTrue((DIAG_ROOT / "origin_plot_data" / f"{name}_origin.csv").exists())


if __name__ == "__main__":
    unittest.main()

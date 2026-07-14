from __future__ import annotations

import csv
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
FIGURE_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune"


class TestPointPillarsTrainingPlotData(unittest.TestCase):
    def test_figures_and_plot_data_exist(self) -> None:
        xy_figures = {
            "02_loss_curve",
            "03_lr_schedule",
            "04_pretrained_vs_finetuned_ap",
            "06_training_runtime_summary",
        }
        all_figures = {
            "01_training_pipeline",
            "02_loss_curve",
            "03_lr_schedule",
            "04_pretrained_vs_finetuned_ap",
            "05_checkpoint_eval_trend",
            "06_training_runtime_summary",
            "07_training_to_deployment_loop",
        }
        for figure in all_figures:
            for suffix in ("png", "svg", "pdf"):
                self.assertTrue((FIGURE_ROOT / f"{figure}.{suffix}").exists(), figure)
        for figure in xy_figures:
            plot_csv = REPORT_ROOT / "plot_data" / f"{figure}.csv"
            origin_csv = REPORT_ROOT / "origin_plot_data" / f"{figure}_origin.csv"
            self.assertTrue(plot_csv.exists(), figure)
            self.assertTrue(origin_csv.exists(), figure)
            with plot_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0, figure)


if __name__ == "__main__":
    unittest.main()

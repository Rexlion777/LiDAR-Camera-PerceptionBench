from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"
FIGURE_ROOT = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures" / "training_finetune"


class TestExpandedHoldoutEvalComparison(unittest.TestCase):
    def test_holdout_comparison_outputs_exist(self) -> None:
        for suffix in ("csv", "md", "json"):
            self.assertTrue((REPORT_ROOT / f"expanded_holdout_eval_comparison.{suffix}").exists())
        for ext in ("png", "svg", "pdf"):
            self.assertTrue((FIGURE_ROOT / f"11_holdout_ap_comparison.{ext}").exists())

    def test_holdout_csv_has_expected_rows_and_scope(self) -> None:
        with (REPORT_ROOT / "expanded_holdout_eval_comparison.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        names = {row["model_name"] for row in rows}
        self.assertEqual(
            names,
            {
                "pretrained_baseline",
                "subset_finetune_200_50_epoch3",
                "expanded_finetune_1000_200_epoch3",
            },
        )
        for row in rows:
            self.assertEqual(row["eval_scope"], "500-frame-holdout")
            self.assertEqual(row["full_val"], "False")

    def test_holdout_json_captures_main_result(self) -> None:
        payload = json.loads((REPORT_ROOT / "expanded_holdout_eval_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["holdout_split"], "holdout_eval_500")
        self.assertEqual(payload["holdout_sample_count"], 500)
        self.assertFalse(payload["full_val"])
        mean_map = payload["mean_ap_3d_moderate"]
        self.assertGreater(mean_map["expanded_finetune_1000_200_epoch3"], mean_map["subset_finetune_200_50_epoch3"])


if __name__ == "__main__":
    unittest.main()

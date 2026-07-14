from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "expanded_splits"


class TestExpandedSplitDistribution(unittest.TestCase):
    def test_split_files_and_manifest_exist(self) -> None:
        required = (
            "split_manifest.json",
            "split_distribution_summary.csv",
            "split_distribution_report.md",
            "train_1000.txt",
            "val_200.txt",
            "holdout_eval_500.txt",
            "train_2000.txt",
            "val_500.txt",
            "holdout_eval_1000.txt",
        )
        for name in required:
            self.assertTrue((SPLIT_ROOT / name).exists(), name)

    def test_manifest_and_overlap_stats_exist(self) -> None:
        payload = json.loads((SPLIT_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["seed"], 42)
        self.assertIn("splits", payload)
        self.assertIn("explicit_overlaps", payload)
        self.assertEqual(payload["splits"]["train_1000"]["sample_count"], 1000)
        self.assertEqual(payload["splits"]["val_200"]["sample_count"], 200)
        self.assertEqual(payload["splits"]["holdout_eval_500"]["sample_count"], 500)
        self.assertEqual(payload["explicit_overlaps"]["train_1000__val_200"], 0)
        self.assertEqual(payload["explicit_overlaps"]["holdout_eval_500__train_1000"], 0)
        self.assertEqual(payload["explicit_overlaps"]["holdout_eval_500__val_200"], 0)

    def test_summary_csv_has_expected_columns_and_rows(self) -> None:
        with (SPLIT_ROOT / "split_distribution_summary.csv").open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 6)
        required_columns = {
            "split_name",
            "sample_count",
            "car_gt_count",
            "ped_gt_count",
            "cyc_gt_count",
            "near_gt_count",
            "mid_gt_count",
            "far_gt_count",
            "avg_points_per_frame",
            "avg_gt_per_frame",
            "overlap_with_train",
            "overlap_with_val",
            "overlap_with_holdout",
            "seed",
        }
        self.assertTrue(required_columns.issubset(rows[0].keys()))
        rows_by_name = {row["split_name"]: row for row in rows}
        self.assertEqual(int(rows_by_name["train_1000"]["sample_count"]), 1000)
        self.assertEqual(int(rows_by_name["val_200"]["sample_count"]), 200)
        self.assertEqual(int(rows_by_name["holdout_eval_500"]["sample_count"]), 500)


if __name__ == "__main__":
    unittest.main()

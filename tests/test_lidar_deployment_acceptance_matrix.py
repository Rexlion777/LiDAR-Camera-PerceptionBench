import csv
import unittest
from pathlib import Path


class DeploymentAcceptanceMatrixTest(unittest.TestCase):
    def test_matrix_and_tables_exist(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance")
        matrix = root / "perturbation_matrix.csv"
        ap = root / "per_setting_ap.csv"
        health = root / "runtime_health_metrics.csv"
        self.assertTrue(matrix.exists())
        self.assertTrue(ap.exists())
        self.assertTrue(health.exists())

        with matrix.open(encoding="utf-8") as handle:
            matrix_rows = list(csv.DictReader(handle))
        with ap.open(encoding="utf-8") as handle:
            ap_rows = list(csv.DictReader(handle))
        with health.open(encoding="utf-8") as handle:
            health_rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(matrix_rows), 20)
        self.assertGreaterEqual(len(ap_rows), 20)
        self.assertGreaterEqual(len(health_rows), 10)
        for field in [
            "perturbation_type",
            "perturbation_value",
            "frame_count",
            "eval_scope",
            "sampling_mode",
            "skipped",
            "skipped_reason",
            "reduced_sampling",
            "reduced_sampling_reason",
        ]:
            self.assertIn(field, matrix_rows[0])
        for field in ["sampling_mode", "car_ap_3d_moderate", "ped_ap_3d_moderate", "cyc_ap_3d_moderate", "mean_ap_3d_moderate", "delta_mean"]:
            self.assertIn(field, ap_rows[0])
        for field in ["label_free_health_risk", "prediction_count_drift", "score_distribution_drift", "mean_ap_drop"]:
            self.assertIn(field, health_rows[0])

        point_dropout = [row for row in ap_rows if row["perturbation_type"] == "point_dropout"]
        score_threshold = [row for row in ap_rows if row["perturbation_type"] == "postprocess_score_threshold"]
        range_crop = [row for row in ap_rows if row["perturbation_type"] == "range_crop"]
        self.assertGreaterEqual(len(point_dropout), 11)
        self.assertGreaterEqual(len(score_threshold), 12)
        self.assertGreaterEqual(len(range_crop), 10)


if __name__ == "__main__":
    unittest.main()

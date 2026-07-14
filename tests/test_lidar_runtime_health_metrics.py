import csv
import unittest
from pathlib import Path


class RuntimeHealthMetricsTest(unittest.TestCase):
    def test_health_metric_outputs(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance")
        with (root / "runtime_health_metrics.csv").open(encoding="utf-8") as handle:
            health = list(csv.DictReader(handle))
        with (root / "health_metric_correlation.csv").open(encoding="utf-8") as handle:
            corr = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(health), 10)
        self.assertGreaterEqual(len(corr), 5)
        metric_names = {row["metric_name"] for row in corr}
        self.assertIn("prediction_count_drift", metric_names)
        self.assertIn("label_free_health_risk", metric_names)
        for row in health:
            self.assertIn("label_free_health_risk", row)
            self.assertIn("mean_ap_drop", row)
            self.assertIn("prediction_count_drift", row)
            self.assertIn("score_distribution_drift", row)
        for row in corr:
            self.assertIn("n_points_used", row)


if __name__ == "__main__":
    unittest.main()

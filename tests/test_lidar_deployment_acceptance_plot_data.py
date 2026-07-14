import json
import unittest
from pathlib import Path


SUMMARY_OR_SKIPPED = {
    "01_application_scenario_deployment_acceptance",
    "02_acceptance_benchmark_pipeline",
    "03_perturbation_matrix_table",
    "04_deployment_boundary_and_trt_result_card",
    "25_trt_outlier_frame_gallery",
    "28_health_metric_correlation_bar",
    "29_final_acceptance_dashboard",
}


class DeploymentAcceptancePlotDataTest(unittest.TestCase):
    def test_dense_plot_data_and_metadata_exist(self):
        plot_data = Path("reports/lidar_system_algorithm/deployment_acceptance/plot_data")
        origin = Path("reports/lidar_system_algorithm/deployment_acceptance/origin_plot_data")
        metadata = Path("reports/lidar_system_algorithm/deployment_acceptance/plot_data_metadata")
        self.assertTrue(plot_data.exists())
        self.assertTrue(origin.exists())
        self.assertTrue(metadata.exists())

        dense_xy = [
            "05_dropout_ap_curve_setting_level",
            "06_dropout_per_frame_box_count_scatter",
            "07_dropout_score_vs_range_box_scatter",
            "08_dropout_gt_range_detected_missed_scatter",
            "09_dropout_failure_heatmap_range_class",
            "10_range_crop_ap_curve_setting_level",
            "11_range_crop_prediction_range_histogram",
            "12_range_crop_gt_range_recall_scatter",
            "13_yaw_reprojection_shift_per_object_scatter",
            "14_yaw_shift_distribution_boxplot",
            "15_time_offset_center_drift_per_frame_scatter",
            "16_time_offset_drift_distribution_boxplot",
            "17_score_threshold_ap_curve",
            "18_score_threshold_box_count_curve",
            "19_score_distribution_hist_by_threshold",
            "20_score_vs_range_scatter_by_threshold",
            "21_class_distribution_vs_threshold",
            "22_trt_per_frame_latency_scatter",
            "23_trt_latency_cdf",
            "24_trt_topk_center_diff_hist",
            "26_per_frame_health_risk_timeline",
            "27_health_risk_vs_failure_proxy_scatter",
        ]
        for name in dense_xy:
            csv_path = plot_data / f"{name}.csv"
            origin_path = origin / f"{name}.csv"
            meta_path = metadata / f"{name}.json"
            self.assertTrue(csv_path.exists(), name)
            self.assertTrue(origin_path.exists(), name)
            self.assertTrue(meta_path.exists(), name)
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("figure_name"), name)
            self.assertTrue(payload.get("source_csv"), name)
            self.assertFalse(payload.get("skipped"), name)
            self.assertIn("data_level", payload)
            self.assertIn("point_count", payload)
            if name not in SUMMARY_OR_SKIPPED and payload.get("data_level") not in {"setting", "summary", "skipped"}:
                self.assertGreaterEqual(int(payload.get("point_count") or 0), 50, name)


if __name__ == "__main__":
    unittest.main()

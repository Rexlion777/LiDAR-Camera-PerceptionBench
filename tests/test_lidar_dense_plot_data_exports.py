import json
import unittest
from pathlib import Path


FIGURE_NAMES = [f"{index:02d}_{name}" for index, name in [
    (1, "application_scenario_deployment_acceptance"),
    (2, "acceptance_benchmark_pipeline"),
    (3, "perturbation_matrix_table"),
    (4, "deployment_boundary_and_trt_result_card"),
    (5, "dropout_ap_curve_setting_level"),
    (6, "dropout_per_frame_box_count_scatter"),
    (7, "dropout_score_vs_range_box_scatter"),
    (8, "dropout_gt_range_detected_missed_scatter"),
    (9, "dropout_failure_heatmap_range_class"),
    (10, "range_crop_ap_curve_setting_level"),
    (11, "range_crop_prediction_range_histogram"),
    (12, "range_crop_gt_range_recall_scatter"),
    (13, "far_dropout_ap_curve"),
    (14, "far_dropout_failure_by_range_heatmap"),
    (15, "noise_ap_curve"),
    (16, "noise_score_range_scatter"),
    (17, "yaw_reprojection_shift_per_object_scatter"),
    (18, "yaw_shift_distribution_boxplot"),
    (19, "pitch_roll_projection_shift_summary"),
    (20, "translation_projection_shift_summary"),
    (21, "time_offset_center_drift_per_frame_scatter"),
    (22, "time_offset_drift_distribution_boxplot"),
    (23, "score_threshold_ap_curve"),
    (24, "score_threshold_box_count_curve"),
    (25, "score_distribution_hist_by_threshold"),
    (26, "score_vs_range_scatter_by_threshold"),
    (27, "class_distribution_vs_threshold"),
    (28, "topk_maxboxes_perturbation_summary"),
    (29, "nms_threshold_summary"),
    (30, "trt_per_frame_latency_scatter_full"),
    (31, "trt_per_frame_latency_scatter_clipped"),
    (32, "trt_latency_cdf"),
    (33, "trt_topk_center_diff_hist"),
    (34, "trt_topk_center_diff_clipped"),
    (35, "trt_outlier_bev_gallery"),
    (36, "per_frame_health_risk_timeline"),
    (37, "health_risk_vs_failure_proxy_scatter"),
    (38, "health_metric_correlation_bar"),
    (39, "selected_1000_dropout_vs_quick_dense_comparison"),
    (40, "selected_1000_range_crop_comparison"),
    (41, "selected_1000_score_threshold_comparison"),
    (42, "final_acceptance_dashboard"),
]]

SUMMARY_OR_SKIPPED = {
    "01_application_scenario_deployment_acceptance",
    "02_acceptance_benchmark_pipeline",
    "03_perturbation_matrix_table",
    "04_deployment_boundary_and_trt_result_card",
    "13_far_dropout_ap_curve",
    "14_far_dropout_failure_by_range_heatmap",
    "15_noise_ap_curve",
    "16_noise_score_range_scatter",
    "19_pitch_roll_projection_shift_summary",
    "20_translation_projection_shift_summary",
    "28_topk_maxboxes_perturbation_summary",
    "29_nms_threshold_summary",
    "35_trt_outlier_bev_gallery",
    "38_health_metric_correlation_bar",
    "39_selected_1000_dropout_vs_quick_dense_comparison",
    "40_selected_1000_range_crop_comparison",
    "41_selected_1000_score_threshold_comparison",
    "42_final_acceptance_dashboard",
}


class DensePlotDataExportsTest(unittest.TestCase):
    def test_metadata_and_origin_exports_exist(self):
        origin = Path("reports/lidar_system_algorithm/deployment_acceptance/origin_plot_data")
        metadata_dir = Path("reports/lidar_system_algorithm/deployment_acceptance/plot_data_metadata")

        for name in FIGURE_NAMES:
            meta_path = metadata_dir / f"{name}.json"
            self.assertTrue(meta_path.exists(), name)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["figure_name"], name)
            self.assertIn("data_level", meta)
            self.assertIn("point_count", meta)
            if name in SUMMARY_OR_SKIPPED or meta.get("skipped"):
                continue
            source_csv = meta.get("source_csv")
            self.assertTrue(source_csv, meta["figure_name"])
            self.assertTrue(Path(source_csv).exists(), meta["figure_name"])
            origin_csv = origin / Path(source_csv).name
            self.assertTrue(origin_csv.exists(), meta["figure_name"])
            if meta.get("data_level") not in {"setting", "summary", "skipped"}:
                self.assertGreaterEqual(int(meta.get("point_count") or 0), 50, meta["figure_name"])

    def test_dense_setting_curves_have_expected_point_counts(self):
        metadata_dir = Path("reports/lidar_system_algorithm/deployment_acceptance/plot_data_metadata")
        expectations = {
            "05_dropout_ap_curve_setting_level": 11,
            "10_range_crop_ap_curve_setting_level": 10,
            "17_yaw_reprojection_shift_per_object_scatter": 50,
            "23_score_threshold_ap_curve": 12,
            "21_time_offset_center_drift_per_frame_scatter": 50,
        }
        for name, minimum in expectations.items():
            meta = json.loads((metadata_dir / f"{name}.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(int(meta.get("point_count") or 0), minimum, name)


if __name__ == "__main__":
    unittest.main()

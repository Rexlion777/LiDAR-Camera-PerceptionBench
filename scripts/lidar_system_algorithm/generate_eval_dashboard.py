from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_json, write_markdown
from runtime.lidar_system_algorithm.visualization import compose_grid, draw_bar_chart, draw_tradeoff_chart, placeholder_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KITTI eval dashboard and failure-case summary.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm", help="Input report directory.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--version", default="v2", help="Dashboard version. Use v2 for bucketed TRT and GT matcher sections.")
    return parser.parse_args()


def _figure_or_placeholder(path: Path):
    try:
        import cv2
        import numpy as np

        if not path.exists():
            return placeholder_image([f"Missing figure: {path.name}"])
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image if image is not None else placeholder_image([f"Unreadable figure: {path.name}"])
    except Exception:
        return placeholder_image([f"Figure unavailable: {path.name}"])


def main() -> None:
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    official_eval = read_json_or_default(input_dir / "kitti_official_eval.json", {})
    online_latency = read_json_or_default(input_dir / "tensorrt_backbone_head_online_latency.json", {})
    voxel = read_json_or_default(input_dir / "voxelization_ablation.json", {})
    tracking = read_json_or_default(input_dir / "tracking_optimized_summary.json", {})
    trt_wrapper = read_json_or_default(input_dir / "tensorrt_real_sample_wrapper.json", {})
    trt_bucketed = read_json_or_default(input_dir / "tensorrt_bucketed_core_report.json", {})
    trt_bucketed_eval = read_json_or_default(input_dir / "tensorrt_bucketed_kitti_eval.json", {})
    trt_backbone_head = read_json_or_default(input_dir / "tensorrt_backbone_head_only_eval.json", {})
    trt_backbone_head_diff = read_json_or_default(input_dir / "tensorrt_backbone_head_only_diff.json", {})
    trt_backbone_head_final = read_json_or_default(input_dir / "tensorrt_backbone_head_only_final_report.json", {})
    trt_backbone_head_fullval = read_json_or_default(input_dir / "tensorrt_backbone_head_only_fullval_eval.json", {})
    final_report = read_json_or_default(input_dir / "lidar_system_algorithm_final_report.json", {})
    failure_matcher = read_json_or_default(input_dir / "failure_matcher_summary.json", {})
    robustness = read_json_or_default(input_dir / "calibration_sync_robustness.json", {})

    eval_dict = official_eval.get("official_result_dict", {}) if isinstance(official_eval, dict) else {}
    ap_rows = [
        {"stage": "Car_mod_3D_AP_R40", "mean_ms": eval_dict.get("Car_3d/moderate_R40")},
        {"stage": "Ped_mod_3D_AP_R40", "mean_ms": eval_dict.get("Pedestrian_3d/moderate_R40")},
        {"stage": "Cyc_mod_3D_AP_R40", "mean_ms": eval_dict.get("Cyclist_3d/moderate_R40")},
    ]
    pytorch_online = online_latency.get("pytorch_summary", {}) if isinstance(online_latency, dict) else {}
    trt_online = online_latency.get("trt_summary", {}) if isinstance(online_latency, dict) else {}
    tracked_rows = voxel.get("summary_rows", []) if isinstance(voxel, dict) else []
    failure_grid = compose_grid(
        [
            _figure_or_placeholder(figures_dir / "failure_case_gallery.png"),
            _figure_or_placeholder(figures_dir / "fp_fn_by_class.png"),
            _figure_or_placeholder(figures_dir / "fp_fn_by_range.png"),
        ],
        columns=3,
        label_lines=["Failure analysis gallery", "GT matcher supports error attribution only; official AP is reported separately."],
    )

    save_image(figures_dir / "kitti_ap_by_class.png", draw_bar_chart(ap_rows, title="KITTI AP_R40 by Class"))
    save_image(figures_dir / "latency_vs_pillar_count.png", draw_tradeoff_chart(tracked_rows, title="Pillar Size vs Pillar Count / Preprocess Latency"))
    save_image(figures_dir / "failure_cases_bev.png", failure_grid)

    payload = {
        "status": "completed",
        "version": args.version,
        "kitti_ap_summary": {
            "Car_3d/moderate_R40": eval_dict.get("Car_3d/moderate_R40"),
            "Pedestrian_3d/moderate_R40": eval_dict.get("Pedestrian_3d/moderate_R40"),
            "Cyclist_3d/moderate_R40": eval_dict.get("Cyclist_3d/moderate_R40"),
        },
        "latency_summary": {
            "pytorch_online_total_ms_mean": pytorch_online.get("online_total_ms", {}).get("mean"),
            "pytorch_online_total_ms_p50": pytorch_online.get("online_total_ms", {}).get("p50"),
            "pytorch_online_total_ms_p95": pytorch_online.get("online_total_ms", {}).get("p95"),
            "trt_online_total_ms_mean": trt_online.get("online_total_ms", {}).get("mean"),
            "trt_online_total_ms_p50": trt_online.get("online_total_ms", {}).get("p50"),
            "trt_online_total_ms_p95": trt_online.get("online_total_ms", {}).get("p95"),
            "pytorch_backbone_head_ms_mean": pytorch_online.get("backbone_head_ms", {}).get("mean"),
            "trt_backbone_head_ms_mean": trt_online.get("backbone_head_ms", {}).get("mean"),
            "online_speedup": online_latency.get("speedup", {}).get("online_total"),
            "core_speedup": online_latency.get("speedup", {}).get("core_only"),
            "visualization_excluded": online_latency.get("visualization_excluded"),
        },
        "tracking_summary": {
            "average_association_latency_ms": tracking.get("average_association_latency_ms"),
            "average_legacy_association_latency_ms": tracking.get("average_legacy_association_latency_ms"),
            "average_track_count": tracking.get("average_track_count"),
        },
        "tensorrt_wrapper_summary": {
            "status": trt_wrapper.get("status"),
            "mean_trt_core_ms": trt_wrapper.get("mean_trt_core_ms"),
            "mean_pytorch_core_ms": trt_wrapper.get("mean_pytorch_core_ms"),
            "mean_online_total_ms": trt_wrapper.get("mean_online_total_ms"),
            "mean_pytorch_online_total_ms": trt_wrapper.get("mean_pytorch_online_total_ms"),
        },
        "tensorrt_bucketed_summary": {
            "status": trt_bucketed.get("status"),
            "successful_bucket_sizes": trt_bucketed.get("successful_bucket_sizes"),
            "overall_truncation_rate": trt_bucketed.get("overall_truncation_rate"),
            "mean_trt_core_ms": trt_bucketed.get("mean_trt_core_ms"),
            "mean_online_total_ms": trt_bucketed.get("mean_online_total_ms"),
            "bucket_hit_distribution": trt_bucketed.get("bucket_hit_distribution"),
        },
        "tensorrt_bucketed_eval_summary": {
            "status": trt_bucketed_eval.get("status"),
            "ap_delta": trt_bucketed_eval.get("ap_delta"),
            "official_result_dict": trt_bucketed_eval.get("official_result_dict"),
        },
        "tensorrt_backbone_head_only_summary": {
            "status": trt_backbone_head.get("status"),
            "ap_comparison": trt_backbone_head_final.get("ap_comparison"),
            "latency_comparison": trt_backbone_head_final.get("latency_comparison"),
            "same_frame_diff_summary": trt_backbone_head_diff.get("summary"),
            "empty_prediction_file_count": trt_backbone_head.get("empty_prediction_file_count"),
            "invalid_geometry_count": trt_backbone_head_diff.get("summary", {}).get("invalid_geometry_count"),
            "deployment_boundary": trt_backbone_head_final.get("scope"),
            "full_core_blocker": trt_backbone_head_final.get("full_bucketed_trt_blocker"),
            "larger_eval": trt_backbone_head_fullval,
        },
        "failure_analysis_summary": {
            "status": failure_matcher.get("status"),
            "totals": failure_matcher.get("totals"),
            "by_class": failure_matcher.get("by_class"),
            "by_range": failure_matcher.get("by_range"),
        },
        "robustness_summary": {
            "yaw_summary": robustness.get("yaw_summary"),
            "time_offset_summary": robustness.get("time_offset_summary"),
        },
        "failure_case_note": "GT matcher is used for error attribution only and does not replace KITTI official evaluation.",
        "dashboard_figures": {
            "ap_by_class": str(figures_dir / "kitti_ap_by_class.png"),
            "latency_vs_pillar_count": str(figures_dir / "latency_vs_pillar_count.png"),
            "failure_cases_bev": str(figures_dir / "failure_cases_bev.png"),
            "fp_fn_by_class": str(figures_dir / "fp_fn_by_class.png"),
            "fp_fn_by_range": str(figures_dir / "fp_fn_by_range.png"),
            "tensorrt_bucket_latency_vs_capacity": str(figures_dir / "tensorrt_bucket_latency_vs_capacity.png"),
            "tensorrt_bucket_hit_distribution": str(figures_dir / "tensorrt_bucket_hit_distribution.png"),
            "trt_backbone_head_ap_comparison": str(figures_dir / "trt_backbone_head_ap_comparison.png"),
            "trt_backbone_head_latency_comparison": str(figures_dir / "trt_backbone_head_latency_comparison.png"),
            "trt_backbone_head_pipeline_boundary": str(figures_dir / "trt_backbone_head_pipeline_boundary.png"),
            "trt_backbone_head_diff_summary": str(figures_dir / "trt_backbone_head_diff_summary.png"),
            "trt_backbone_head_online_latency_breakdown": str(figures_dir / "trt_backbone_head_online_latency_breakdown.png"),
            "trt_backbone_head_fullval_ap_comparison": str(figures_dir / "trt_backbone_head_fullval_ap_comparison.png"),
            "system_pipeline_final": str(figures_dir / "system_pipeline_final.png"),
        },
    }
    write_json(output_dir / "eval_dashboard.json", payload)
    write_markdown(
        output_dir / "eval_dashboard.md",
        f"""# Eval Dashboard

## AP Summary

- Car moderate 3D AP_R40: `{payload['kitti_ap_summary']['Car_3d/moderate_R40']}`
- Pedestrian moderate 3D AP_R40: `{payload['kitti_ap_summary']['Pedestrian_3d/moderate_R40']}`
- Cyclist moderate 3D AP_R40: `{payload['kitti_ap_summary']['Cyclist_3d/moderate_R40']}`

## Latency Summary

- PyTorch online_total mean/p50/p95 ms: `{payload['latency_summary'].get('pytorch_online_total_ms_mean')}` / `{payload['latency_summary'].get('pytorch_online_total_ms_p50')}` / `{payload['latency_summary'].get('pytorch_online_total_ms_p95')}`
- TRT online_total mean/p50/p95 ms: `{payload['latency_summary'].get('trt_online_total_ms_mean')}` / `{payload['latency_summary'].get('trt_online_total_ms_p50')}` / `{payload['latency_summary'].get('trt_online_total_ms_p95')}`
- PyTorch backbone/head mean ms: `{payload['latency_summary'].get('pytorch_backbone_head_ms_mean')}`
- TRT backbone/head mean ms: `{payload['latency_summary'].get('trt_backbone_head_ms_mean')}`
- core speedup: `{payload['latency_summary'].get('core_speedup')}`
- online speedup: `{payload['latency_summary'].get('online_speedup')}`
- visualization excluded from online path: `{payload['latency_summary'].get('visualization_excluded')}`

## Tracking Summary

- legacy association latency ms: `{payload['tracking_summary']['average_legacy_association_latency_ms']}`
- optimized association latency ms: `{payload['tracking_summary']['average_association_latency_ms']}`
- average track count: `{payload['tracking_summary']['average_track_count']}`

## TensorRT Wrapper

- status: `{payload['tensorrt_wrapper_summary']['status']}`
- TensorRT core mean ms: `{payload['tensorrt_wrapper_summary']['mean_trt_core_ms']}`
- PyTorch core mean ms: `{payload['tensorrt_wrapper_summary']['mean_pytorch_core_ms']}`
- online wrapper latency is reported honestly as preprocessing + TensorRT core + postprocess, not as a full TensorRT detector.

## TensorRT Bucketed Core

- status: `{payload['tensorrt_bucketed_summary']['status']}`
- successful bucket sizes: `{payload['tensorrt_bucketed_summary']['successful_bucket_sizes']}`
- overall truncation rate: `{payload['tensorrt_bucketed_summary']['overall_truncation_rate']}`
- bucket hit distribution: `{payload['tensorrt_bucketed_summary']['bucket_hit_distribution']}`

## Backbone/Head-only TensorRT Deployment

- status: `{payload['tensorrt_backbone_head_only_summary']['status']}`
- AP comparison: `{payload['tensorrt_backbone_head_only_summary']['ap_comparison']}`
- latency comparison: `{payload['tensorrt_backbone_head_only_summary']['latency_comparison']}`
- same-frame diff summary: `{payload['tensorrt_backbone_head_only_summary']['same_frame_diff_summary']}`
- empty prediction files: `{payload['tensorrt_backbone_head_only_summary']['empty_prediction_file_count']}`
- invalid geometry count: `{payload['tensorrt_backbone_head_only_summary']['invalid_geometry_count']}`
- deployment boundary: `{payload['tensorrt_backbone_head_only_summary']['deployment_boundary']}`
- full-core blocker: `{payload['tensorrt_backbone_head_only_summary']['full_core_blocker']}`
- larger eval summary: `{payload['tensorrt_backbone_head_only_summary']['larger_eval'].get('result_scope')}` / `Car {payload['tensorrt_backbone_head_only_summary']['larger_eval'].get('trt_backbone_head_only', {}).get('official_result_dict', {}).get('Car_3d/moderate_R40')}` / `Ped {payload['tensorrt_backbone_head_only_summary']['larger_eval'].get('trt_backbone_head_only', {}).get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}` / `Cyc {payload['tensorrt_backbone_head_only_summary']['larger_eval'].get('trt_backbone_head_only', {}).get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`

## Failure Analysis

- matcher status: `{payload['failure_analysis_summary']['status']}`
- TP / FP / FN: `{payload['failure_analysis_summary'].get('totals')}`
- GT matcher is used for analysis only; official AP still comes from the KITTI evaluator.

## Robustness

- yaw perturbation summary: `{payload['robustness_summary']['yaw_summary']}`
- time offset proxy summary: `{payload['robustness_summary']['time_offset_summary']}`

## Limitations

- TensorRT scope remains core-only; voxelization and NMS are still outside the engine.
- Bucketed fixed-shape capacity analysis must not be described as a full dynamic TensorRT detector.
- Backbone/head-only TensorRT is a valid milestone; full TensorRT detector is still blocked by the full bucketed core path.
- GT matcher is an analysis tool, not an official evaluator replacement.
""",
    )
    print(f"Saved eval dashboard: {output_dir / 'eval_dashboard.json'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_json, write_markdown
from runtime.lidar_system_algorithm.visualization import draw_bar_chart, placeholder_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final LiDAR system algorithm report.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = read_json_or_default(input_dir / "kitti_official_eval.json", {})
    wrapper_v2 = read_json_or_default(input_dir / "wrapper_pytorch_core_eval_v2.json", {})
    trt_200 = read_json_or_default(input_dir / "tensorrt_backbone_head_only_eval.json", {})
    trt_diff = read_json_or_default(input_dir / "tensorrt_backbone_head_only_diff.json", {})
    larger_eval = read_json_or_default(input_dir / "tensorrt_backbone_head_only_fullval_eval.json", {})
    online_latency = read_json_or_default(input_dir / "tensorrt_backbone_head_online_latency.json", {})
    tracking = read_json_or_default(input_dir / "tracking_optimized_summary.json", {})
    robustness = read_json_or_default(input_dir / "calibration_sync_robustness.json", {})
    failure = read_json_or_default(input_dir / "failure_analysis_report.json", {})
    trt_final = read_json_or_default(input_dir / "tensorrt_backbone_head_only_final_report.json", {})
    trt_200_latency = trt_200.get("latency_summary", {})
    trt_200_pytorch_core = trt_200_latency.get("pytorch_core_ms", {})
    trt_200_trt_core = trt_200_latency.get("trt_core_ms", {})
    larger_trt_latency = larger_eval.get("trt_backbone_head_only", {}).get("latency_summary", {})

    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    system_pipeline_final = figures_dir / "system_pipeline_final.png"
    fullval_ap_fig = figures_dir / "trt_backbone_head_fullval_ap_comparison.png"
    save_image(
        system_pipeline_final,
        placeholder_image(
            [
                "LiDAR system milestone",
                "KITTI / OpenPCDet / PointPillars baseline",
                "wrapper parity -> backbone/head-only TRT -> profiling / tracking / robustness / failure analysis",
                "PyTorch: preprocessing, voxelization, VFE, scatter",
                "TensorRT: BEV backbone, dense head",
                "OpenPCDet native: post_processing, export, evaluator integration",
            ],
            width=1400,
            height=800,
        ),
    )
    larger_baseline = larger_eval.get("baseline", {}).get("official_result_dict", {})
    larger_wrapper = larger_eval.get("wrapper_pytorch_core", {}).get("official_result_dict", {})
    larger_trt = larger_eval.get("trt_backbone_head_only", {}).get("official_result_dict", {})
    save_image(
        fullval_ap_fig,
        draw_bar_chart(
            [
                {"stage": "Baseline_Car", "mean_ms": larger_baseline.get("Car_3d/moderate_R40")},
                {"stage": "Wrapper_Car", "mean_ms": larger_wrapper.get("Car_3d/moderate_R40")},
                {"stage": "TRT_Car", "mean_ms": larger_trt.get("Car_3d/moderate_R40")},
                {"stage": "Baseline_Ped", "mean_ms": larger_baseline.get("Pedestrian_3d/moderate_R40")},
                {"stage": "Wrapper_Ped", "mean_ms": larger_wrapper.get("Pedestrian_3d/moderate_R40")},
                {"stage": "TRT_Ped", "mean_ms": larger_trt.get("Pedestrian_3d/moderate_R40")},
                {"stage": "Baseline_Cyc", "mean_ms": larger_baseline.get("Cyclist_3d/moderate_R40")},
                {"stage": "Wrapper_Cyc", "mean_ms": larger_wrapper.get("Cyclist_3d/moderate_R40")},
                {"stage": "TRT_Cyc", "mean_ms": larger_trt.get("Cyclist_3d/moderate_R40")},
            ],
            title="Larger-scope AP_R40 Comparison",
        ),
    )

    payload = {
        "status": "completed",
        "project_goal": "Turn the LiDAR/PointPillars demo into a system-algorithm runtime with honest deployment scope.",
        "dataset_and_eval_scope": {
            "baseline_full_val": baseline.get("official_result_dict", {}),
            "slice_200": wrapper_v2.get("official_result_dict", {}),
            "larger_eval": larger_eval,
        },
        "wrapper_parity_fix": {
            "root_cause": trt_final.get("wrapper_parity_root_cause"),
            "wrapper_eval_v2": wrapper_v2.get("official_result_dict", {}),
        },
        "backbone_head_only_trt": {
            "scope": trt_final.get("scope"),
            "eval_200": trt_200.get("official_result_dict", {}),
            "same_frame_diff": trt_diff.get("summary", {}),
            "latency_200": trt_200_latency,
            "larger_eval": larger_eval.get("trt_backbone_head_only", {}),
        },
        "online_latency": online_latency,
        "tracking": tracking,
        "robustness": robustness,
        "failure_analysis": failure,
        "full_bucketed_trt_blocker": trt_final.get("full_bucketed_trt_blocker"),
        "resume_safe_claims": trt_final.get("resume_safe_claims"),
        "forbidden_claims": trt_final.get("forbidden_claims"),
        "figures": {
            "system_pipeline_final": str(system_pipeline_final),
            "ap_comparison": str(figures_dir / "trt_backbone_head_ap_comparison.png"),
            "latency_comparison": str(figures_dir / "trt_backbone_head_latency_comparison.png"),
            "online_latency_breakdown": str(figures_dir / "trt_backbone_head_online_latency_breakdown.png"),
            "fullval_ap_comparison": str(fullval_ap_fig),
        },
    }

    json_path = output_dir / "lidar_system_algorithm_final_report.json"
    md_path = output_dir / "lidar_system_algorithm_final_report.md"
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# LiDAR System Algorithm Final Report\n\n"
        "## 1. Project Goal\n\n"
        "- Build a reproducible LiDAR 3D perception runtime around KITTI / OpenPCDet / PointPillars and extend it with profiling, tracking, robustness, failure analysis, and honest deployment milestones.\n\n"
        "## 2. Dataset and Evaluation Scope\n\n"
        f"- Baseline full-val official AP is available in `kitti_official_eval.json`.\n"
        f"- 200-frame slice wrapper AP: `{wrapper_v2.get('official_result_dict', {}).get('Car_3d/moderate_R40')}` / `{wrapper_v2.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}` / `{wrapper_v2.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`\n"
        f"- Larger eval scope: `{larger_eval.get('result_scope')}`; skipped reason: `{larger_eval.get('skipped_reason')}`\n\n"
        "## 3. PyTorch Baseline / Wrapper Parity\n\n"
        f"- Wrapper parity root cause: `{trt_final.get('wrapper_parity_root_cause', {}).get('root_cause')}`\n"
        "- Fix kept native post_processing, generate_prediction_dicts, thresholds, NMS config and official evaluator untouched.\n\n"
        "## 4. Backbone/Head-only TensorRT\n\n"
        f"- 200-frame AP: `{trt_200.get('official_result_dict', {}).get('Car_3d/moderate_R40')}` / `{trt_200.get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}` / `{trt_200.get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`\n"
        f"- 200-frame core latency mean/p50/p95: PyTorch `{trt_200_pytorch_core.get('mean')}` / `{trt_200_pytorch_core.get('p50')}` / `{trt_200_pytorch_core.get('p95')}` ms vs TRT `{trt_200_trt_core.get('mean')}` / `{trt_200_trt_core.get('p50')}` / `{trt_200_trt_core.get('p95')}` ms\n"
        f"- Larger eval AP: `{larger_eval.get('trt_backbone_head_only', {}).get('official_result_dict', {}).get('Car_3d/moderate_R40')}` / `{larger_eval.get('trt_backbone_head_only', {}).get('official_result_dict', {}).get('Pedestrian_3d/moderate_R40')}` / `{larger_eval.get('trt_backbone_head_only', {}).get('official_result_dict', {}).get('Cyclist_3d/moderate_R40')}`\n\n"
        f"- Larger eval core latency mean/p50/p95: PyTorch `{larger_trt_latency.get('pytorch_core_ms', {}).get('mean')}` / `{larger_trt_latency.get('pytorch_core_ms', {}).get('p50')}` / `{larger_trt_latency.get('pytorch_core_ms', {}).get('p95')}` ms vs TRT `{larger_trt_latency.get('trt_core_ms', {}).get('mean')}` / `{larger_trt_latency.get('trt_core_ms', {}).get('p50')}` / `{larger_trt_latency.get('trt_core_ms', {}).get('p95')}` ms\n\n"
        "## 5. Same-frame Diff\n\n"
        f"- Summary: `{trt_diff.get('summary')}`\n\n"
        "## 6. Online End-to-end Profiling\n\n"
        f"- PyTorch online_total mean/p50/p95: `{online_latency.get('pytorch_summary', {}).get('online_total_ms', {}).get('mean')}` / `{online_latency.get('pytorch_summary', {}).get('online_total_ms', {}).get('p50')}` / `{online_latency.get('pytorch_summary', {}).get('online_total_ms', {}).get('p95')}`\n"
        f"- TRT online_total mean/p50/p95: `{online_latency.get('trt_summary', {}).get('online_total_ms', {}).get('mean')}` / `{online_latency.get('trt_summary', {}).get('online_total_ms', {}).get('p50')}` / `{online_latency.get('trt_summary', {}).get('online_total_ms', {}).get('p95')}`\n"
        "- Visualization remains excluded from online latency.\n\n"
        "## 7. Tracking / Robustness / Failure Analysis\n\n"
        f"- Tracking optimized association latency: `{tracking.get('average_association_latency_ms')}` ms\n"
        f"- Calibration/time-sync robustness report: `{robustness.get('status')}`\n"
        f"- Failure analysis report: `{failure.get('status')}`\n\n"
        "## 8. Boundaries\n\n"
        "- Valid claim: backbone/head-only TensorRT milestone with official AP parity and core latency gain.\n"
        "- Invalid claim: full TensorRT PointPillars detector.\n"
        f"- Full bucketed TRT blocker still exists: `{trt_final.get('full_bucketed_trt_blocker')}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()

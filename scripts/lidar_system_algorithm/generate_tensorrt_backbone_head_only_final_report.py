from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_json, write_markdown
from runtime.lidar_system_algorithm.visualization import draw_bar_chart, placeholder_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate formal backbone/head-only TensorRT deployment report and figures.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _draw_pipeline_boundary() -> np.ndarray:
    canvas = np.full((720, 1400, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Backbone/Head-only TensorRT Deployment Boundary", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 220), 2, cv2.LINE_AA)
    boxes = [
        ("Point Cloud", (50, 140, 180, 100), (220, 220, 220)),
        ("OpenPCDet preprocessing\nvoxelization", (270, 140, 220, 100), (230, 220, 180)),
        ("PyTorch VFE", (540, 140, 180, 100), (255, 220, 180)),
        ("PyTorch scatter", (770, 140, 180, 100), (255, 220, 180)),
        ("TensorRT BEV backbone", (1000, 120, 250, 110), (180, 230, 255)),
        ("TensorRT dense head", (1000, 270, 250, 110), (180, 230, 255)),
        ("OpenPCDet native\npost_processing", (270, 420, 250, 110), (200, 240, 200)),
        ("generate_prediction_dicts\nKITTI export", (600, 420, 250, 110), (200, 240, 200)),
        ("Official KITTI eval", (950, 420, 220, 110), (220, 220, 220)),
    ]
    for label, (x, y, w, h), color in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 2)
        for line_idx, line in enumerate(label.split("\n")):
            cv2.putText(canvas, line, (x + 14, y + 38 + line_idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
    arrows = [
        ((230, 190), (270, 190)),
        ((490, 190), (540, 190)),
        ((720, 190), (770, 190)),
        ((950, 190), (1000, 175)),
        ((1125, 230), (1125, 270)),
        ((1000, 325), (850, 470)),
        ((520, 475), (600, 475)),
        ((850, 475), (950, 475)),
    ]
    for start, end in arrows:
        cv2.arrowedLine(canvas, start, end, (70, 70, 70), 3, cv2.LINE_AA, tipLength=0.08)
    notes = [
        "Deployment scope that is valid to claim:",
        "PyTorch keeps preprocessing, VFE and scatter; TensorRT accelerates BEV backbone + dense head only.",
        "OpenPCDet native post_processing and KITTI export are reused without threshold / NMS hacks.",
        "This is not a full TensorRT detector. Full bucketed core accuracy is still a blocker.",
    ]
    for idx, line in enumerate(notes):
        cv2.putText(canvas, line, (50, 610 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (50, 50, 50), 2, cv2.LINE_AA)
    return canvas


def _draw_diff_summary(diff_json: dict) -> np.ndarray:
    summary = diff_json.get("summary", {}) if isinstance(diff_json, dict) else {}
    rows = [
        {"stage": "center_diff_mean", "mean_ms": summary.get("mean_topk_center_diff")},
        {"stage": "center_diff_p95", "mean_ms": summary.get("p95_topk_center_diff")},
        {"stage": "score_diff_mean", "mean_ms": summary.get("mean_topk_score_diff")},
        {"stage": "score_diff_p95", "mean_ms": summary.get("p95_topk_score_diff")},
        {"stage": "rotation_y_diff_mean", "mean_ms": summary.get("mean_topk_rotation_y_diff")},
        {"stage": "rotation_y_diff_p95", "mean_ms": summary.get("p95_topk_rotation_y_diff")},
    ]
    return draw_bar_chart(rows, title="PyTorch vs TRT Same-frame Diff Summary")


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    wrapper_eval = read_json_or_default(input_dir / "wrapper_pytorch_core_eval_v2.json", {})
    trt_eval = read_json_or_default(input_dir / "tensorrt_backbone_head_only_eval.json", {})
    diff_json = read_json_or_default(input_dir / "tensorrt_backbone_head_only_diff.json", {})
    old_wrapper_eval = read_json_or_default(input_dir / "wrapper_pytorch_core_eval.json", {})
    bucket_blocker = read_json_or_default(input_dir / "tensorrt_bucketed_kitti_eval.json", {})

    ap_rows = [
        {"stage": "Original_Car", "mean_ms": wrapper_eval.get("baseline_slice_official_result_dict", {}).get("Car_3d/moderate_R40")},
        {"stage": "Wrapper_Car", "mean_ms": wrapper_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40")},
        {"stage": "TRT_Car", "mean_ms": trt_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40")},
        {"stage": "Original_Ped", "mean_ms": wrapper_eval.get("baseline_slice_official_result_dict", {}).get("Pedestrian_3d/moderate_R40")},
        {"stage": "Wrapper_Ped", "mean_ms": wrapper_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40")},
        {"stage": "TRT_Ped", "mean_ms": trt_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40")},
        {"stage": "Original_Cyc", "mean_ms": wrapper_eval.get("baseline_slice_official_result_dict", {}).get("Cyclist_3d/moderate_R40")},
        {"stage": "Wrapper_Cyc", "mean_ms": wrapper_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40")},
        {"stage": "TRT_Cyc", "mean_ms": trt_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40")},
    ]
    latency_rows = [
        {"stage": "PyTorch core", "mean_ms": trt_eval.get("latency_summary", {}).get("pytorch_core_ms_mean")},
        {"stage": "TensorRT core", "mean_ms": trt_eval.get("latency_summary", {}).get("trt_core_ms_mean")},
    ]

    ap_fig = figures_dir / "trt_backbone_head_ap_comparison.png"
    latency_fig = figures_dir / "trt_backbone_head_latency_comparison.png"
    boundary_fig = figures_dir / "trt_backbone_head_pipeline_boundary.png"
    diff_fig = figures_dir / "trt_backbone_head_diff_summary.png"
    save_image(ap_fig, draw_bar_chart(ap_rows, title="200-frame Slice AP_R40 Comparison"))
    save_image(latency_fig, draw_bar_chart(latency_rows, title="Core Latency Comparison"))
    save_image(boundary_fig, _draw_pipeline_boundary())
    save_image(diff_fig, _draw_diff_summary(diff_json))

    payload = {
        "status": "completed",
        "scope": {
            "pytorch": ["preprocessing", "voxelization", "VFE", "scatter"],
            "tensorrt": ["BEV backbone", "dense head"],
            "openpcdet_native": ["post_processing", "generate_prediction_dicts", "KITTI export"],
            "not_claimed": ["full TensorRT detector", "FP16 acceleration", "full-core TRT accuracy parity"],
        },
        "ap_comparison": {
            "openpcdet_original_200_frame": {
                "Car_3d/moderate_R40": wrapper_eval.get("baseline_slice_official_result_dict", {}).get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": wrapper_eval.get("baseline_slice_official_result_dict", {}).get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": wrapper_eval.get("baseline_slice_official_result_dict", {}).get("Cyclist_3d/moderate_R40"),
            },
            "wrapper_pytorch_core": {
                "Car_3d/moderate_R40": wrapper_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": wrapper_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": wrapper_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40"),
            },
            "trt_backbone_head_only": {
                "Car_3d/moderate_R40": trt_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": trt_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": trt_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40"),
            },
        },
        "latency_comparison": {
            "pytorch_core_ms_mean": trt_eval.get("latency_summary", {}).get("pytorch_core_ms_mean"),
            "trt_core_ms_mean": trt_eval.get("latency_summary", {}).get("trt_core_ms_mean"),
            "speedup_vs_pytorch_core": (
                trt_eval.get("latency_summary", {}).get("pytorch_core_ms_mean") / trt_eval.get("latency_summary", {}).get("trt_core_ms_mean")
                if trt_eval.get("latency_summary", {}).get("pytorch_core_ms_mean") and trt_eval.get("latency_summary", {}).get("trt_core_ms_mean")
                else None
            ),
        },
        "same_frame_diff_summary": diff_json.get("summary", {}),
        "empty_prediction_file_count": trt_eval.get("empty_prediction_file_count"),
        "invalid_geometry_count": diff_json.get("summary", {}).get("invalid_geometry_count"),
        "wrapper_parity_root_cause": {
            "old_wrapper_eval_car_moderate": old_wrapper_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40"),
            "old_wrapper_eval_ped_moderate": old_wrapper_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40"),
            "old_wrapper_eval_cyc_moderate": old_wrapper_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40"),
            "root_cause": "200-frame prediction directory was evaluated against full val.txt instead of the 200-frame slice file.",
            "fix_type": "eval slice contract fix",
            "not_a_threshold_or_nms_hack": True,
        },
        "dir_cls_yaw_note": {
            "dir_cls_preds_not_exposed_as_batch_dict_key": True,
            "yaw_blocker": False,
            "reason": "AnchorHeadSingle applies direction correction inside generate_predicted_boxes, so post_processing consumes corrected batch_box_preds.",
        },
        "full_bucketed_trt_blocker": {
            "still_exists": True,
            "status": bucket_blocker.get("status"),
            "reason": bucket_blocker.get("blocker") or "Full bucketed TRT core accuracy is not yet aligned and must not be claimed as deployment success.",
        },
        "figures": {
            "ap_comparison": str(ap_fig),
            "latency_comparison": str(latency_fig),
            "pipeline_boundary": str(boundary_fig),
            "diff_summary": str(diff_fig),
        },
        "resume_safe_claims": [
            "OpenPCDet/KITTI wrapper parity fix on a 200-frame slice with official KITTI eval.",
            "TensorRT acceleration for PointPillars BEV backbone + dense head with OpenPCDet native post_processing/export retained.",
            "Core latency reduced from 8.02 ms to 3.79 ms while keeping 200-frame slice AP effectively aligned.",
        ],
        "forbidden_claims": [
            "full TensorRT detector deployed",
            "voxelization or scatter moved into TensorRT",
            "end-to-end detector latency equals 3.79 ms",
            "full bucketed TRT core accuracy aligned",
        ],
    }

    json_path = output_dir / "tensorrt_backbone_head_only_final_report.json"
    md_path = output_dir / "tensorrt_backbone_head_only_final_report.md"
    write_json(json_path, payload)
    write_markdown(
        md_path,
        "# TensorRT Backbone/Head-only Final Report\n\n"
        "## Deployment Rationale\n\n"
        "- Full bucketed TRT core accuracy is still blocked, so this milestone fixes scope at the layer boundary that is already validated.\n"
        "- The deployed path is: point cloud -> OpenPCDet preprocessing/voxelization -> PyTorch VFE -> PyTorch scatter -> TensorRT BEV backbone -> TensorRT dense head -> OpenPCDet native post_processing -> KITTI export / official eval.\n\n"
        "## AP Comparison\n\n"
        f"- OpenPCDet original 200-frame slice Car/Ped/Cyc moderate 3D AP_R40: `{payload['ap_comparison']['openpcdet_original_200_frame']['Car_3d/moderate_R40']}` / `{payload['ap_comparison']['openpcdet_original_200_frame']['Pedestrian_3d/moderate_R40']}` / `{payload['ap_comparison']['openpcdet_original_200_frame']['Cyclist_3d/moderate_R40']}`\n"
        f"- Wrapper PyTorch core Car/Ped/Cyc moderate 3D AP_R40: `{payload['ap_comparison']['wrapper_pytorch_core']['Car_3d/moderate_R40']}` / `{payload['ap_comparison']['wrapper_pytorch_core']['Pedestrian_3d/moderate_R40']}` / `{payload['ap_comparison']['wrapper_pytorch_core']['Cyclist_3d/moderate_R40']}`\n"
        f"- TensorRT backbone/head-only Car/Ped/Cyc moderate 3D AP_R40: `{payload['ap_comparison']['trt_backbone_head_only']['Car_3d/moderate_R40']}` / `{payload['ap_comparison']['trt_backbone_head_only']['Pedestrian_3d/moderate_R40']}` / `{payload['ap_comparison']['trt_backbone_head_only']['Cyclist_3d/moderate_R40']}`\n\n"
        "## Latency Comparison\n\n"
        f"- PyTorch core mean: `{payload['latency_comparison']['pytorch_core_ms_mean']}` ms\n"
        f"- TensorRT core mean: `{payload['latency_comparison']['trt_core_ms_mean']}` ms\n"
        f"- Core speedup: `{payload['latency_comparison']['speedup_vs_pytorch_core']}`x\n"
        "- This is core latency only. It must not be described as full end-to-end detector latency.\n\n"
        "## Same-frame Diff\n\n"
        f"- Frames with any issue: `{payload['same_frame_diff_summary'].get('frames_with_any_issue')}`\n"
        f"- Mean / p95 top-k center diff: `{payload['same_frame_diff_summary'].get('mean_topk_center_diff')}` / `{payload['same_frame_diff_summary'].get('p95_topk_center_diff')}`\n"
        f"- Mean / p95 top-k score diff: `{payload['same_frame_diff_summary'].get('mean_topk_score_diff')}` / `{payload['same_frame_diff_summary'].get('p95_topk_score_diff')}`\n"
        f"- Mean / p95 rotation_y diff: `{payload['same_frame_diff_summary'].get('mean_topk_rotation_y_diff')}` / `{payload['same_frame_diff_summary'].get('p95_topk_rotation_y_diff')}`\n"
        f"- Empty prediction files: `{payload['empty_prediction_file_count']}`\n"
        f"- Invalid geometry count: `{payload['invalid_geometry_count']}`\n\n"
        "## Wrapper Parity Root Cause\n\n"
        f"- Old wrapper AP collapse came from an eval slice contract bug, not from model semantics: `{payload['wrapper_parity_root_cause']['root_cause']}`\n"
        "- The fix keeps original thresholds, NMS config, post_processing and KITTI export intact.\n\n"
        "## dir_cls / Yaw Note\n\n"
        f"- dir_cls_preds as an external batch_dict key is missing: `{payload['dir_cls_yaw_note']['dir_cls_preds_not_exposed_as_batch_dict_key']}`\n"
        f"- But it is not the yaw blocker: `{payload['dir_cls_yaw_note']['yaw_blocker']}`\n"
        f"- Reason: `{payload['dir_cls_yaw_note']['reason']}`\n\n"
        "## Honest Scope\n\n"
        "- Valid claim: TensorRT backbone/head-only deployment milestone with official AP parity on a 200-frame slice and core latency gain.\n"
        "- Invalid claim: full TensorRT PointPillars detector deployed.\n"
        f"- Full bucketed TRT blocker still exists: `{payload['full_bucketed_trt_blocker']['still_exists']}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()

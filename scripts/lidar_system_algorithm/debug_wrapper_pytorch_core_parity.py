from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.failure_matcher import read_kitti_objects
from runtime.lidar_system_algorithm.report_schema import read_json_or_default, write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.tensorrt_debug_utils import numeric_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenPCDet baseline vs wrapper_pytorch_core predictions frame by frame.")
    parser.add_argument("--baseline-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt")
    parser.add_argument("--wrapper-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_wrapper_pytorch_core_v2")
    parser.add_argument("--baseline-eval-json", default="reports/lidar_system_algorithm/kitti_official_eval.json")
    parser.add_argument("--wrapper-eval-json", default="reports/lidar_system_algorithm/wrapper_pytorch_core_eval_v2.json")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _object_to_dict(obj) -> dict:
    return {
        "class_name": obj.class_name,
        "score": obj.score,
        "center": [obj.location_camera_xyz[0], obj.location_camera_xyz[1], obj.location_camera_xyz[2]],
        "dimensions_hwl": [obj.dimensions_hwl[0], obj.dimensions_hwl[1], obj.dimensions_hwl[2]],
        "rotation_y": obj.rotation_y,
        "bbox_2d": list(obj.bbox),
    }


def _safe_mean(values) -> float | None:
    seq = [float(value) for value in values if value is not None]
    return float(np.mean(seq)) if seq else None


def _summarize_objects(objects: list) -> dict:
    scores = [obj.score for obj in objects if obj.score is not None]
    class_counter = Counter(obj.class_name for obj in objects)
    invalid_geometry = 0
    nonfinite = 0
    for obj in objects:
        values = list(obj.location_camera_xyz) + list(obj.dimensions_hwl) + [obj.rotation_y]
        if not all(np.isfinite(values)):
            nonfinite += 1
        if any(value <= 0 for value in obj.dimensions_hwl):
            invalid_geometry += 1
    return {
        "box_count": len(objects),
        "per_class_box_count": dict(class_counter),
        "score_summary": numeric_summary(scores),
        "invalid_geometry_count": invalid_geometry,
        "nonfinite_count": nonfinite,
    }


def _topk_signature(objects: list, topk: int) -> list[dict]:
    ranked = sorted(objects, key=lambda item: float(item.score or 0.0), reverse=True)[:topk]
    return [_object_to_dict(obj) for obj in ranked]


def _class_count(counter_dict: dict, class_name: str) -> int:
    return int(counter_dict.get(class_name, 0))


def _diagnose(frame_row: dict) -> list[str]:
    issues: list[str] = []
    class_delta = frame_row["class_distribution_l1"]
    score_delta = frame_row["score_mean_delta"]
    yaw_delta = frame_row["topk_rotation_y_abs_diff_mean"]
    center_delta = frame_row["topk_center_l2_diff_mean"]
    if frame_row["baseline_line_count"] != frame_row["wrapper_line_count"]:
        issues.append("nms_mismatch")
    if frame_row["baseline_empty_file"] != frame_row["wrapper_empty_file"]:
        issues.append("kitti_export_mismatch")
    if class_delta and class_delta > max(3, 0.25 * max(frame_row["baseline_box_count"], 1)):
        issues.append("class_mapping_mismatch")
    if score_delta is not None and abs(score_delta) > 0.2:
        issues.append("score_sigmoid_or_normalization_mismatch")
    if yaw_delta is not None and yaw_delta > 0.5:
        issues.append("missing_dir_cls_or_yaw_correction")
    if center_delta is not None and center_delta > 5.0:
        issues.append("box_decode_mismatch")
    if frame_row["sample_id_mismatch"]:
        issues.append("frame_id_mismatch")
    if frame_row["wrapper_invalid_geometry_count"] > frame_row["baseline_invalid_geometry_count"]:
        issues.append("padded_batch_dict_contract_mismatch")
    return issues


def main() -> None:
    args = parse_args()
    baseline_pred_dir = _resolve(args.baseline_pred_dir)
    wrapper_pred_dir = _resolve(args.wrapper_pred_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "wrapper_pytorch_core_parity_report.json"
    report_md = output_dir / "wrapper_pytorch_core_parity_report.md"
    frame_csv = output_dir / "wrapper_pytorch_core_frame_diff.csv"
    topk_csv = output_dir / "wrapper_pytorch_core_topk_diff.csv"

    wrapper_eval = read_json_or_default(_resolve(args.wrapper_eval_json), {})
    baseline_eval = read_json_or_default(_resolve(args.baseline_eval_json), {})
    frame_ids = []
    if isinstance(wrapper_eval, dict):
        frame_ids = [str(row["frame_id"]) for row in wrapper_eval.get("frame_rows_preview", []) if row.get("frame_id")]
    if not frame_ids:
        frame_ids = sorted(path.stem for path in wrapper_pred_dir.glob("*.txt"))
    if isinstance(wrapper_eval, dict) and wrapper_eval.get("frame_count", 0) and wrapper_eval["frame_count"] > len(frame_ids):
        frame_ids = sorted(path.stem for path in wrapper_pred_dir.glob("*.txt"))

    frame_rows: list[dict] = []
    topk_rows: list[dict] = []
    diagnosis_counter = Counter()
    for frame_id in frame_ids:
        baseline_path = baseline_pred_dir / f"{frame_id}.txt"
        wrapper_path = wrapper_pred_dir / f"{frame_id}.txt"
        baseline_objects = read_kitti_objects(baseline_path, is_prediction=True) if baseline_path.exists() else []
        wrapper_objects = read_kitti_objects(wrapper_path, is_prediction=True) if wrapper_path.exists() else []
        baseline_summary = _summarize_objects(baseline_objects)
        wrapper_summary = _summarize_objects(wrapper_objects)
        baseline_topk = _topk_signature(baseline_objects, args.topk)
        wrapper_topk = _topk_signature(wrapper_objects, args.topk)
        topk_center_diffs = []
        topk_dim_diffs = []
        topk_yaw_diffs = []
        topk_class_mismatches = 0
        for rank in range(max(len(baseline_topk), len(wrapper_topk))):
            a = baseline_topk[rank] if rank < len(baseline_topk) else None
            b = wrapper_topk[rank] if rank < len(wrapper_topk) else None
            if a and b:
                topk_center_diffs.append(float(np.linalg.norm(np.asarray(a["center"]) - np.asarray(b["center"]))))
                topk_dim_diffs.append(float(np.linalg.norm(np.asarray(a["dimensions_hwl"]) - np.asarray(b["dimensions_hwl"]))))
                topk_yaw_diffs.append(abs(float(a["rotation_y"]) - float(b["rotation_y"])))
                if a["class_name"] != b["class_name"]:
                    topk_class_mismatches += 1
            topk_rows.append(
                {
                    "sample_id": frame_id,
                    "rank": rank,
                    "baseline_class_name": None if a is None else a["class_name"],
                    "wrapper_class_name": None if b is None else b["class_name"],
                    "baseline_score": None if a is None else a["score"],
                    "wrapper_score": None if b is None else b["score"],
                    "baseline_center": None if a is None else json.dumps(a["center"]),
                    "wrapper_center": None if b is None else json.dumps(b["center"]),
                    "baseline_dimensions_hwl": None if a is None else json.dumps(a["dimensions_hwl"]),
                    "wrapper_dimensions_hwl": None if b is None else json.dumps(b["dimensions_hwl"]),
                    "baseline_rotation_y": None if a is None else a["rotation_y"],
                    "wrapper_rotation_y": None if b is None else b["rotation_y"],
                    "baseline_bbox_2d": None if a is None else json.dumps(a["bbox_2d"]),
                    "wrapper_bbox_2d": None if b is None else json.dumps(b["bbox_2d"]),
                }
            )

        baseline_counter = baseline_summary["per_class_box_count"]
        wrapper_counter = wrapper_summary["per_class_box_count"]
        class_distribution_l1 = sum(abs(_class_count(baseline_counter, cls) - _class_count(wrapper_counter, cls)) for cls in ["Car", "Pedestrian", "Cyclist"])
        frame_row = {
            "sample_id": frame_id,
            "baseline_box_count": baseline_summary["box_count"],
            "wrapper_box_count": wrapper_summary["box_count"],
            "baseline_car_count": _class_count(baseline_counter, "Car"),
            "baseline_pedestrian_count": _class_count(baseline_counter, "Pedestrian"),
            "baseline_cyclist_count": _class_count(baseline_counter, "Cyclist"),
            "wrapper_car_count": _class_count(wrapper_counter, "Car"),
            "wrapper_pedestrian_count": _class_count(wrapper_counter, "Pedestrian"),
            "wrapper_cyclist_count": _class_count(wrapper_counter, "Cyclist"),
            "baseline_score_mean": baseline_summary["score_summary"]["mean"],
            "wrapper_score_mean": wrapper_summary["score_summary"]["mean"],
            "baseline_score_p50": baseline_summary["score_summary"]["p50"],
            "wrapper_score_p50": wrapper_summary["score_summary"]["p50"],
            "baseline_score_p95": baseline_summary["score_summary"]["p95"],
            "wrapper_score_p95": wrapper_summary["score_summary"]["p95"],
            "score_mean_delta": None if baseline_summary["score_summary"]["mean"] is None or wrapper_summary["score_summary"]["mean"] is None else wrapper_summary["score_summary"]["mean"] - baseline_summary["score_summary"]["mean"],
            "topk_score_abs_diff_mean": _safe_mean([
                abs(float(a["score"]) - float(b["score"]))
                for a, b in zip(baseline_topk, wrapper_topk)
                if a.get("score") is not None and b.get("score") is not None
            ]),
            "topk_class_mismatch_count": topk_class_mismatches,
            "topk_center_l2_diff_mean": _safe_mean(topk_center_diffs),
            "topk_dimensions_l2_diff_mean": _safe_mean(topk_dim_diffs),
            "topk_rotation_y_abs_diff_mean": _safe_mean(topk_yaw_diffs),
            "baseline_line_count": baseline_summary["box_count"],
            "wrapper_line_count": wrapper_summary["box_count"],
            "baseline_empty_file": baseline_summary["box_count"] == 0,
            "wrapper_empty_file": wrapper_summary["box_count"] == 0,
            "class_distribution_l1": class_distribution_l1,
            "baseline_invalid_geometry_count": baseline_summary["invalid_geometry_count"],
            "wrapper_invalid_geometry_count": wrapper_summary["invalid_geometry_count"],
            "baseline_nonfinite_count": baseline_summary["nonfinite_count"],
            "wrapper_nonfinite_count": wrapper_summary["nonfinite_count"],
            "sample_id_mismatch": baseline_path.stem != wrapper_path.stem,
        }
        issues = _diagnose(frame_row)
        for issue in issues:
            diagnosis_counter[issue] += 1
        frame_row["diagnosis"] = "|".join(issues)
        frame_rows.append(frame_row)

    wrapper_result = wrapper_eval.get("official_result_dict", {}) if isinstance(wrapper_eval, dict) else {}
    baseline_result = baseline_eval.get("official_result_dict", {}) if isinstance(baseline_eval, dict) else {}
    payload = {
        "status": "completed",
        "frame_count": len(frame_rows),
        "baseline_pred_dir": str(baseline_pred_dir),
        "wrapper_pred_dir": str(wrapper_pred_dir),
        "diagnosis_counts": dict(diagnosis_counter),
        "ap_summary": {
            "baseline": {
                "Car_3d/moderate_R40": baseline_result.get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": baseline_result.get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": baseline_result.get("Cyclist_3d/moderate_R40"),
            },
            "wrapper": {
                "Car_3d/moderate_R40": wrapper_result.get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": wrapper_result.get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": wrapper_result.get("Cyclist_3d/moderate_R40"),
            },
        },
        "summary": {
            "baseline_box_count_mean": _safe_mean([row["baseline_box_count"] for row in frame_rows]),
            "wrapper_box_count_mean": _safe_mean([row["wrapper_box_count"] for row in frame_rows]),
            "topk_center_l2_diff_mean": _safe_mean([row["topk_center_l2_diff_mean"] for row in frame_rows]),
            "topk_rotation_y_abs_diff_mean": _safe_mean([row["topk_rotation_y_abs_diff_mean"] for row in frame_rows]),
            "frames_with_any_issue": int(sum(1 for row in frame_rows if row["diagnosis"])),
        },
        "frame_rows_preview": frame_rows[:20],
    }
    write_json(report_json, payload)
    write_csv(report_dir := frame_csv, frame_rows, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["sample_id"])
    write_csv(topk_csv, topk_rows, fieldnames=list(topk_rows[0].keys()) if topk_rows else ["sample_id", "rank"])
    write_markdown(
        report_md,
        "# Wrapper PyTorch Core Parity\n\n"
        f"- Frame count: `{payload['frame_count']}`\n"
        f"- Diagnosis counts: `{payload['diagnosis_counts']}`\n"
        f"- Baseline Car/Ped/Cyc moderate AP_R40: `{payload['ap_summary']['baseline']['Car_3d/moderate_R40']}` / `{payload['ap_summary']['baseline']['Pedestrian_3d/moderate_R40']}` / `{payload['ap_summary']['baseline']['Cyclist_3d/moderate_R40']}`\n"
        f"- Wrapper Car/Ped/Cyc moderate AP_R40: `{payload['ap_summary']['wrapper']['Car_3d/moderate_R40']}` / `{payload['ap_summary']['wrapper']['Pedestrian_3d/moderate_R40']}` / `{payload['ap_summary']['wrapper']['Cyclist_3d/moderate_R40']}`\n"
        f"- Mean top-k center diff: `{payload['summary']['topk_center_l2_diff_mean']}`\n"
        f"- Mean top-k rotation_y diff: `{payload['summary']['topk_rotation_y_abs_diff_mean']}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(report_json)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Compare wrapper_pytorch_core vs TensorRT backbone/head-only outputs frame by frame.")
    parser.add_argument("--pytorch-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_wrapper_pytorch_core_v2")
    parser.add_argument("--trt-pred-dir", default="projects/lidar_system_algorithm/results/kitti_eval_txt_trt_backbone_head_only")
    parser.add_argument("--pytorch-eval-json", default="reports/lidar_system_algorithm/wrapper_pytorch_core_eval_v2.json")
    parser.add_argument("--trt-eval-json", default="reports/lidar_system_algorithm/tensorrt_backbone_head_only_eval.json")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _safe_mean(values) -> float | None:
    seq = [float(value) for value in values if value is not None]
    return float(np.mean(seq)) if seq else None


def _percentile(values, pct: float) -> float | None:
    seq = [float(value) for value in values if value is not None]
    return float(np.percentile(np.asarray(seq, dtype=np.float64), pct)) if seq else None


def _safe_max(values) -> float | None:
    seq = [float(value) for value in values if value is not None]
    return float(max(seq)) if seq else None


def _object_to_dict(obj) -> dict:
    return {
        "class_name": obj.class_name,
        "score": obj.score,
        "center": [obj.location_camera_xyz[0], obj.location_camera_xyz[1], obj.location_camera_xyz[2]],
        "dimensions_hwl": [obj.dimensions_hwl[0], obj.dimensions_hwl[1], obj.dimensions_hwl[2]],
        "rotation_y": obj.rotation_y,
        "bbox_2d": list(obj.bbox),
    }


def _summarize_objects(objects: list) -> dict:
    scores = [obj.score for obj in objects if obj.score is not None]
    class_counter = Counter(obj.class_name for obj in objects)
    invalid_geometry = 0
    nonfinite = 0
    score_all_one = bool(scores) and all(abs(float(score) - 1.0) < 1e-6 for score in scores)
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
        "score_all_one": score_all_one,
        "class_collapse": len(class_counter) <= 1 and len(objects) > 0,
    }


def _topk_signature(objects: list, topk: int) -> list[dict]:
    ranked = sorted(objects, key=lambda item: float(item.score or 0.0), reverse=True)[:topk]
    return [_object_to_dict(obj) for obj in ranked]


def _class_count(counter_dict: dict, class_name: str) -> int:
    return int(counter_dict.get(class_name, 0))


def main() -> None:
    args = parse_args()
    pytorch_pred_dir = _resolve(args.pytorch_pred_dir)
    trt_pred_dir = _resolve(args.trt_pred_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "tensorrt_backbone_head_only_diff.json"
    md_path = output_dir / "tensorrt_backbone_head_only_diff.md"
    frame_csv = output_dir / "tensorrt_backbone_head_only_frame_diff.csv"
    topk_csv = output_dir / "tensorrt_backbone_head_only_topk_diff.csv"

    pytorch_eval = read_json_or_default(_resolve(args.pytorch_eval_json), {})
    trt_eval = read_json_or_default(_resolve(args.trt_eval_json), {})
    frame_ids = sorted({path.stem for path in pytorch_pred_dir.glob("*.txt")} | {path.stem for path in trt_pred_dir.glob("*.txt")})

    frame_rows: list[dict] = []
    topk_rows: list[dict] = []
    center_diffs = []
    score_diffs = []
    yaw_diffs = []
    invalid_geometry_total = 0
    empty_prediction_files = 0
    class_distribution_l1_values = []
    per_frame_center_mean_diffs = []

    for frame_id in frame_ids:
        py_path = pytorch_pred_dir / f"{frame_id}.txt"
        trt_path = trt_pred_dir / f"{frame_id}.txt"
        py_objects = read_kitti_objects(py_path, is_prediction=True) if py_path.exists() else []
        trt_objects = read_kitti_objects(trt_path, is_prediction=True) if trt_path.exists() else []
        py_summary = _summarize_objects(py_objects)
        trt_summary = _summarize_objects(trt_objects)
        py_topk = _topk_signature(py_objects, args.topk)
        trt_topk = _topk_signature(trt_objects, args.topk)

        topk_center_diffs = []
        topk_dim_diffs = []
        topk_yaw_diffs = []
        topk_score_diffs = []
        topk_class_mismatches = 0

        for rank in range(max(len(py_topk), len(trt_topk))):
            a = py_topk[rank] if rank < len(py_topk) else None
            b = trt_topk[rank] if rank < len(trt_topk) else None
            if a and b:
                topk_center_diffs.append(float(np.linalg.norm(np.asarray(a["center"]) - np.asarray(b["center"]))))
                topk_dim_diffs.append(float(np.linalg.norm(np.asarray(a["dimensions_hwl"]) - np.asarray(b["dimensions_hwl"]))))
                topk_yaw_diffs.append(abs(float(a["rotation_y"]) - float(b["rotation_y"])))
                if a["score"] is not None and b["score"] is not None:
                    topk_score_diffs.append(abs(float(a["score"]) - float(b["score"])))
                if a["class_name"] != b["class_name"]:
                    topk_class_mismatches += 1
            topk_rows.append(
                {
                    "sample_id": frame_id,
                    "rank": rank,
                    "pytorch_class_name": None if a is None else a["class_name"],
                    "trt_class_name": None if b is None else b["class_name"],
                    "pytorch_score": None if a is None else a["score"],
                    "trt_score": None if b is None else b["score"],
                    "pytorch_center": None if a is None else json.dumps(a["center"]),
                    "trt_center": None if b is None else json.dumps(b["center"]),
                    "pytorch_dimensions_hwl": None if a is None else json.dumps(a["dimensions_hwl"]),
                    "trt_dimensions_hwl": None if b is None else json.dumps(b["dimensions_hwl"]),
                    "pytorch_rotation_y": None if a is None else a["rotation_y"],
                    "trt_rotation_y": None if b is None else b["rotation_y"],
                    "pytorch_bbox_2d": None if a is None else json.dumps(a["bbox_2d"]),
                    "trt_bbox_2d": None if b is None else json.dumps(b["bbox_2d"]),
                }
            )

        py_counter = py_summary["per_class_box_count"]
        trt_counter = trt_summary["per_class_box_count"]
        class_distribution_l1 = sum(abs(_class_count(py_counter, cls) - _class_count(trt_counter, cls)) for cls in ["Car", "Pedestrian", "Cyclist"])
        class_distribution_l1_values.append(class_distribution_l1)
        invalid_geometry_total += trt_summary["invalid_geometry_count"]
        if trt_summary["box_count"] == 0:
            empty_prediction_files += 1

        row = {
            "sample_id": frame_id,
            "pytorch_box_count": py_summary["box_count"],
            "trt_box_count": trt_summary["box_count"],
            "pytorch_car_count": _class_count(py_counter, "Car"),
            "pytorch_pedestrian_count": _class_count(py_counter, "Pedestrian"),
            "pytorch_cyclist_count": _class_count(py_counter, "Cyclist"),
            "trt_car_count": _class_count(trt_counter, "Car"),
            "trt_pedestrian_count": _class_count(trt_counter, "Pedestrian"),
            "trt_cyclist_count": _class_count(trt_counter, "Cyclist"),
            "pytorch_score_min": py_summary["score_summary"]["min"],
            "pytorch_score_max": py_summary["score_summary"]["max"],
            "pytorch_score_mean": py_summary["score_summary"]["mean"],
            "pytorch_score_p50": py_summary["score_summary"]["p50"],
            "pytorch_score_p95": py_summary["score_summary"]["p95"],
            "trt_score_min": trt_summary["score_summary"]["min"],
            "trt_score_max": trt_summary["score_summary"]["max"],
            "trt_score_mean": trt_summary["score_summary"]["mean"],
            "trt_score_p50": trt_summary["score_summary"]["p50"],
            "trt_score_p95": trt_summary["score_summary"]["p95"],
            "topk_score_abs_diff_mean": _safe_mean(topk_score_diffs),
            "topk_center_l2_diff_mean": _safe_mean(topk_center_diffs),
            "topk_dimensions_l2_diff_mean": _safe_mean(topk_dim_diffs),
            "topk_rotation_y_abs_diff_mean": _safe_mean(topk_yaw_diffs),
            "topk_class_mismatch_count": topk_class_mismatches,
            "nms_box_count_diff": abs(py_summary["box_count"] - trt_summary["box_count"]),
            "empty_file_flag": int(trt_summary["box_count"] == 0),
            "invalid_geometry_count": trt_summary["invalid_geometry_count"],
            "nonfinite_count": trt_summary["nonfinite_count"],
            "class_collapse_flag": int(trt_summary["class_collapse"]),
            "score_all_one_flag": int(trt_summary["score_all_one"]),
            "class_distribution_l1": class_distribution_l1,
        }
        frame_rows.append(row)
        center_diffs.extend(topk_center_diffs)
        score_diffs.extend(topk_score_diffs)
        yaw_diffs.extend(topk_yaw_diffs)
        if topk_center_diffs:
            per_frame_center_mean_diffs.append({"frame_id": frame_id, "value": float(np.mean(topk_center_diffs))})

    center_p50 = _percentile(center_diffs, 50.0)
    center_p95 = _percentile(center_diffs, 95.0)
    center_p99 = _percentile(center_diffs, 99.0)
    center_max = _safe_max(center_diffs)
    outlier_threshold = None
    if center_p99 is not None:
        outlier_threshold = max(1.0, center_p99 * 5.0)
    elif center_p95 is not None:
        outlier_threshold = max(1.0, center_p95 * 10.0)
    outlier_frames = [
        row["frame_id"]
        for row in per_frame_center_mean_diffs
        if outlier_threshold is not None and row["value"] > outlier_threshold
    ]
    outlier_rows = [
        {
            "frame_id": row["sample_id"],
            "reason": "large_center_diff_outlier"
            if outlier_threshold is not None and (row["topk_center_l2_diff_mean"] or 0.0) > outlier_threshold
            else "class_or_nms_mismatch"
        }
        for row in frame_rows
        if row["sample_id"] in outlier_frames or row["nms_box_count_diff"] or row["topk_class_mismatch_count"]
    ]

    payload = {
        "status": "completed",
        "frame_count": len(frame_rows),
        "pytorch_pred_dir": str(pytorch_pred_dir),
        "trt_pred_dir": str(trt_pred_dir),
        "ap_comparison": {
            "wrapper_pytorch_core": {
                "Car_3d/moderate_R40": pytorch_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": pytorch_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": pytorch_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40"),
            },
            "trt_backbone_head_only": {
                "Car_3d/moderate_R40": trt_eval.get("official_result_dict", {}).get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": trt_eval.get("official_result_dict", {}).get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": trt_eval.get("official_result_dict", {}).get("Cyclist_3d/moderate_R40"),
            },
        },
        "summary": {
            "frames_with_any_issue": int(sum(1 for row in frame_rows if row["nms_box_count_diff"] or row["topk_class_mismatch_count"] or row["class_collapse_flag"] or row["score_all_one_flag"] or row["nonfinite_count"] or row["invalid_geometry_count"])),
            "mean_topk_center_diff": _safe_mean(center_diffs),
            "count_used_for_mean": len(center_diffs),
            "count_used_for_p95": len(center_diffs),
            "p50_topk_center_diff": center_p50,
            "p95_topk_center_diff": center_p95,
            "p99_topk_center_diff": center_p99,
            "max_topk_center_diff": center_max,
            "mean_topk_score_diff": _safe_mean(score_diffs),
            "p95_topk_score_diff": _percentile(score_diffs, 95.0),
            "mean_topk_rotation_y_diff": _safe_mean(yaw_diffs),
            "p95_topk_rotation_y_diff": _percentile(yaw_diffs, 95.0),
            "empty_prediction_file_count": empty_prediction_files,
            "invalid_geometry_count": invalid_geometry_total,
            "class_distribution_match": bool(sum(class_distribution_l1_values) == 0),
            "score_distribution_match": abs((trt_eval.get("score_summary", {}) or {}).get("mean", 0.0) - (pytorch_eval.get("score_summary", {}) or {}).get("mean", 0.0)) < 1e-3,
            "score_all_one_frame_count": int(sum(row["score_all_one_flag"] for row in frame_rows)),
            "class_collapse_frame_count": int(sum(row["class_collapse_flag"] for row in frame_rows)),
            "outlier_frame_ids": outlier_frames,
            "outlier_reason": "mean is computed over all top-k pair diffs; a few large center-diff outliers can raise mean above p95",
        },
        "outlier_rows": outlier_rows[:20],
        "frame_rows_preview": frame_rows[:20],
    }

    write_json(json_path, payload)
    write_csv(frame_csv, frame_rows, fieldnames=list(frame_rows[0].keys()) if frame_rows else ["sample_id"])
    write_csv(topk_csv, topk_rows, fieldnames=list(topk_rows[0].keys()) if topk_rows else ["sample_id", "rank"])
    write_markdown(
        md_path,
        "# TensorRT Backbone/Head-only Same-frame Diff\n\n"
        f"- Frame count: `{payload['frame_count']}`\n"
        f"- Frames with any issue: `{payload['summary']['frames_with_any_issue']}`\n"
        f"- Mean / p95 top-k center diff: `{payload['summary']['mean_topk_center_diff']}` / `{payload['summary']['p95_topk_center_diff']}`\n"
        f"- p50 / p99 / max top-k center diff: `{payload['summary']['p50_topk_center_diff']}` / `{payload['summary']['p99_topk_center_diff']}` / `{payload['summary']['max_topk_center_diff']}`\n"
        f"- Count used for mean/p95: `{payload['summary']['count_used_for_mean']}` / `{payload['summary']['count_used_for_p95']}`\n"
        f"- Mean / p95 top-k score diff: `{payload['summary']['mean_topk_score_diff']}` / `{payload['summary']['p95_topk_score_diff']}`\n"
        f"- Mean / p95 rotation_y diff: `{payload['summary']['mean_topk_rotation_y_diff']}` / `{payload['summary']['p95_topk_rotation_y_diff']}`\n"
        f"- Empty prediction files: `{payload['summary']['empty_prediction_file_count']}`\n"
        f"- Invalid geometry count: `{payload['summary']['invalid_geometry_count']}`\n"
        f"- Class distribution match: `{payload['summary']['class_distribution_match']}`\n"
        f"- Score distribution match: `{payload['summary']['score_distribution_match']}`\n",
    )
    print(json.dumps({"status": "completed", "report": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()

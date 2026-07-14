from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .failure_matcher import (
    DEFAULT_IOU_THRESHOLDS,
    SUPPORTED_CLASSES,
    KittiObject,
    bev_iou,
    greedy_match_objects,
    read_kitti_objects,
)
from .report_schema import read_json_or_default


def _safe_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return value
    return value


def numeric_summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    seq = [float(value) for value in values if value is not None]
    if not seq:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    seq.sort()

    def percentile(pct: float) -> float:
        if len(seq) == 1:
            return seq[0]
        position = (len(seq) - 1) * pct / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return seq[lower]
        mix = position - lower
        return seq[lower] * (1.0 - mix) + seq[upper] * mix

    return {
        "count": len(seq),
        "min": seq[0],
        "max": seq[-1],
        "mean": sum(seq) / len(seq),
        "p50": percentile(50.0),
        "p95": percentile(95.0),
    }


def summarize_prediction_dir(pred_dir: Path, frame_ids: Iterable[str] | None = None) -> dict:
    frame_ids = list(frame_ids) if frame_ids is not None else sorted(path.stem for path in pred_dir.glob("*.txt"))
    files = [pred_dir / f"{frame_id}.txt" for frame_id in frame_ids]
    existing = [path for path in files if path.exists()]
    empty = []
    total_boxes = 0
    score_values: list[float] = []
    class_counter = Counter()
    per_file_rows = []
    for path in existing:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        stripped = raw_text.strip()
        if not stripped:
            empty.append(path.name)
            per_file_rows.append(
                {
                    "frame_id": path.stem,
                    "file_exists": True,
                    "line_count": 0,
                    "box_count": 0,
                    "empty_file": True,
                    "token_count_set": "",
                    "class_names": "",
                }
            )
            continue
        objs = read_kitti_objects(path, is_prediction=True)
        token_counts = sorted({len(line.split()) for line in raw_text.splitlines() if line.strip()})
        file_classes = Counter(obj.class_name for obj in objs)
        per_file_rows.append(
            {
                "frame_id": path.stem,
                "file_exists": True,
                "line_count": len([line for line in raw_text.splitlines() if line.strip()]),
                "box_count": len(objs),
                "empty_file": False,
                "token_count_set": ",".join(str(item) for item in token_counts),
                "class_names": ",".join(sorted(file_classes)),
            }
        )
        total_boxes += len(objs)
        for obj in objs:
            class_counter[obj.class_name] += 1
            if obj.score is not None:
                score_values.append(float(obj.score))
    return {
        "frame_count": len(frame_ids),
        "prediction_file_count": len(existing),
        "missing_prediction_file_count": len(files) - len(existing),
        "empty_prediction_file_count": len(empty),
        "empty_prediction_files_preview": empty[:10],
        "total_box_count": total_boxes,
        "per_class_box_count": dict(class_counter),
        "score_summary": numeric_summary(score_values),
        "per_file_rows": per_file_rows,
    }


def summarize_wrapper_frame_rows(frame_rows: list[dict], prefix: str) -> dict:
    box_key = f"{prefix}_box_count"
    score_key = "score_mean" if prefix == "trt" else None
    box_counts = [row.get(box_key, 0) for row in frame_rows]
    scores = [row.get(score_key) for row in frame_rows] if score_key else []
    return {
        "frame_count": len(frame_rows),
        "prediction_file_count": 0,
        "empty_prediction_file_count": 0,
        "total_box_count": int(sum(int(value or 0) for value in box_counts)),
        "per_class_box_count": {},
        "score_summary": numeric_summary(scores),
        "box_count_summary": numeric_summary(box_counts),
    }


def build_route_summaries(
    baseline_eval: dict,
    wrapper_eval: dict,
    bucket_report: dict,
    bucket_eval: dict,
    raw_diff: dict,
    baseline_pred_dir: Path,
    trt_pred_dir: Path,
) -> list[dict]:
    frame_rows = bucket_report.get("frame_rows", []) if isinstance(bucket_report, dict) else []
    route_a_stats = summarize_prediction_dir(baseline_pred_dir)
    route_b_stats = summarize_wrapper_frame_rows(frame_rows, "pytorch")
    route_c_stats = summarize_prediction_dir(trt_pred_dir)
    route_d_rows = raw_diff.get("rows", []) if isinstance(raw_diff, dict) else []
    route_d_total = len(route_d_rows)
    route_d_nonfinite = sum(
        1
        for row in route_d_rows
        if any(
            math.isnan(float(row.get(key))) or math.isinf(float(row.get(key)))
            for key in ("cls_abs_diff_mean", "box_abs_diff_mean")
            if row.get(key) is not None
        )
    )
    baseline_dict = baseline_eval.get("official_result_dict", {}) if isinstance(baseline_eval, dict) else {}
    trt_dict = bucket_eval.get("official_result_dict", {}) if isinstance(bucket_eval, dict) else {}
    wrapper_dict = wrapper_eval.get("official_result_dict", {}) if isinstance(wrapper_eval, dict) else {}
    return [
        {
            "route": "A_openpcdet_original",
            "frame_count": baseline_eval.get("frame_count", route_a_stats["frame_count"]),
            "prediction_file_count": route_a_stats["prediction_file_count"],
            "empty_prediction_file_count": route_a_stats["empty_prediction_file_count"],
            "total_box_count": route_a_stats["total_box_count"],
            "per_class_box_count": route_a_stats["per_class_box_count"],
            "score_summary": route_a_stats["score_summary"],
            "official_eval_status": baseline_eval.get("status"),
            "ap_summary": {
                "Car_3d/moderate_R40": baseline_dict.get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": baseline_dict.get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": baseline_dict.get("Cyclist_3d/moderate_R40"),
            },
            "blocker": None,
        },
        {
            "route": "B_wrapper_pytorch_core",
            "frame_count": wrapper_eval.get("frame_count", route_b_stats["frame_count"]),
            "prediction_file_count": wrapper_eval.get("prediction_file_count", route_b_stats["prediction_file_count"]),
            "empty_prediction_file_count": wrapper_eval.get("empty_prediction_file_count", route_b_stats["empty_prediction_file_count"]),
            "total_box_count": wrapper_eval.get("total_box_count", route_b_stats["total_box_count"]),
            "per_class_box_count": wrapper_eval.get("per_class_box_count", route_b_stats["per_class_box_count"]),
            "score_summary": wrapper_eval.get("score_summary", route_b_stats["score_summary"]),
            "official_eval_status": wrapper_eval.get("official_eval_status", wrapper_eval.get("status", "skipped")),
            "ap_summary": {
                "Car_3d/moderate_R40": wrapper_dict.get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": wrapper_dict.get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": wrapper_dict.get("Cyclist_3d/moderate_R40"),
            },
            "blocker": wrapper_eval.get("blocker"),
        },
        {
            "route": "C_wrapper_trt_core",
            "frame_count": bucket_eval.get("frame_count", route_c_stats["frame_count"]),
            "prediction_file_count": route_c_stats["prediction_file_count"],
            "empty_prediction_file_count": route_c_stats["empty_prediction_file_count"],
            "total_box_count": route_c_stats["total_box_count"],
            "per_class_box_count": route_c_stats["per_class_box_count"],
            "score_summary": route_c_stats["score_summary"],
            "official_eval_status": bucket_eval.get("status"),
            "ap_summary": {
                "Car_3d/moderate_R40": trt_dict.get("Car_3d/moderate_R40"),
                "Pedestrian_3d/moderate_R40": trt_dict.get("Pedestrian_3d/moderate_R40"),
                "Cyclist_3d/moderate_R40": trt_dict.get("Cyclist_3d/moderate_R40"),
            },
            "blocker": bucket_eval.get("blocker"),
        },
        {
            "route": "D_wrapper_trt_core_no_export",
            "frame_count": route_d_total,
            "prediction_file_count": 0,
            "empty_prediction_file_count": 0,
            "total_box_count": 0,
            "per_class_box_count": {},
            "score_summary": {},
            "official_eval_status": "skipped",
            "ap_summary": {},
            "blocker": f"Raw tensor diff only. {route_d_nonfinite}/{route_d_total} sampled frames show non-finite or infinite raw-output diffs.",
        },
    ]


def analyze_raw_tensor_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    analysis_rows = []
    severe_frames = []
    for row in rows:
        cls_mean = _safe_float(row.get("cls_abs_diff_mean"))
        box_mean = _safe_float(row.get("box_abs_diff_mean"))
        py_min = _safe_float(row.get("py_cls_min"))
        py_max = _safe_float(row.get("py_cls_max"))
        trt_min = _safe_float(row.get("trt_cls_min"))
        trt_max = _safe_float(row.get("trt_cls_max"))
        issue = []
        if cls_mean is not None and abs(cls_mean) > 1000:
            issue.append("cls_magnitude_mismatch")
        if box_mean is not None and math.isinf(box_mean):
            issue.append("box_non_finite_diff")
        if trt_min is not None and py_min is not None and abs(trt_min) > max(1000.0, abs(py_min) * 100.0):
            issue.append("trt_logit_scale_exploded")
        if trt_max is not None and py_max is not None and abs(trt_max) > max(1000.0, abs(py_max) * 100.0):
            issue.append("trt_peak_exploded")
        severity = "severe" if issue else "aligned_or_mild"
        if severity == "severe":
            severe_frames.append(row.get("frame_id"))
        analysis_rows.append(
            {
                "frame_id": row.get("frame_id"),
                "bucket_size": row.get("bucket_size"),
                "pillar_count": row.get("pillar_count"),
                "cls_abs_diff_mean": cls_mean,
                "box_abs_diff_mean": box_mean,
                "py_cls_min": py_min,
                "py_cls_max": py_max,
                "trt_cls_min": trt_min,
                "trt_cls_max": trt_max,
                "severity": severity,
                "suspected_issue": "|".join(issue),
            }
        )
    summary = {
        "sampled_frame_count": len(rows),
        "severe_frame_count": len(severe_frames),
        "severe_frames": severe_frames,
        "suspected_layer": "TRT raw output / tensor semantic mismatch" if severe_frames else "no obvious raw-output blocker in sampled frames",
    }
    return analysis_rows, summary


def analyze_decode_nms_rows(frame_rows: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    zero_trt_frames = []
    nonzero_py_frames = []
    for row in frame_rows:
        py_count = int(row.get("pytorch_box_count", 0) or 0)
        trt_count = int(row.get("trt_box_count", 0) or 0)
        issue = []
        if py_count > 0 and trt_count == 0:
            issue.append("trt_post_nms_empty")
            zero_trt_frames.append(row.get("frame_id"))
        if py_count > 0:
            nonzero_py_frames.append(row.get("frame_id"))
        rows.append(
            {
                "frame_id": row.get("frame_id"),
                "bucket_size": row.get("selected_bucket_size"),
                "full_pillar_count": row.get("full_pillar_count"),
                "padding_pillars": row.get("padding_pillars"),
                "pytorch_box_count": py_count,
                "trt_box_count": trt_count,
                "topk_center_diff_mean": row.get("topk_center_diff_mean"),
                "pytorch_core_ms": row.get("pytorch_core_ms"),
                "trt_core_ms": row.get("trt_core_ms"),
                "issue": "|".join(issue),
            }
        )
    summary = {
        "frame_count": len(frame_rows),
        "pytorch_nonzero_box_frames": len(nonzero_py_frames),
        "trt_zero_box_frames": len(zero_trt_frames),
        "suspected_layer": "decode / postprocess receives unusable TRT tensors" if zero_trt_frames else "decode / NMS did not show empty-TRT symptom",
    }
    return rows, summary


def audit_prediction_export(pred_dir: Path, frame_ids: Iterable[str] | None = None) -> tuple[list[dict], dict]:
    summary = summarize_prediction_dir(pred_dir, frame_ids=frame_ids)
    audit_rows = []
    invalid_token_files = 0
    valid_class_only_files = 0
    line_field_issue_files = 0
    for row in summary["per_file_rows"]:
        token_set = str(row.get("token_count_set", ""))
        token_ok = token_set in ("", "16")
        class_names = [name for name in str(row.get("class_names", "")).split(",") if name]
        classes_ok = all(name in SUPPORTED_CLASSES for name in class_names)
        if not token_ok and row["line_count"] > 0:
            invalid_token_files += 1
        if not classes_ok:
            line_field_issue_files += 1
        if classes_ok:
            valid_class_only_files += 1
        audit_rows.append(
            {
                "frame_id": row["frame_id"],
                "file_exists": row["file_exists"],
                "empty_file": row["empty_file"],
                "line_count": row["line_count"],
                "box_count": row["box_count"],
                "token_count_set": token_set,
                "token_count_expected": "16",
                "class_names": row["class_names"],
                "class_names_valid": classes_ok,
                "line_fields_valid": token_ok,
            }
        )
    export_summary = {
        "frame_count": summary["frame_count"],
        "prediction_file_count": summary["prediction_file_count"],
        "missing_prediction_file_count": summary["missing_prediction_file_count"],
        "empty_prediction_file_count": summary["empty_prediction_file_count"],
        "invalid_token_file_count": invalid_token_files,
        "valid_class_only_file_count": valid_class_only_files,
        "line_field_issue_file_count": line_field_issue_files,
        "empty_prediction_files_preview": summary["empty_prediction_files_preview"],
        "score_summary": summary["score_summary"],
        "direct_symptom": "all prediction files empty" if summary["prediction_file_count"] > 0 and summary["empty_prediction_file_count"] == summary["prediction_file_count"] else "prediction files contain at least some boxes",
    }
    return audit_rows, export_summary


def _collect_frame_ids(label_dir: Path, pred_dir: Path) -> list[str]:
    return sorted({path.stem for path in label_dir.glob("*.txt")} | {path.stem for path in pred_dir.glob("*.txt")})


def _duplicate_like_fp_count(gt_objects: list[KittiObject], unmatched_preds: list[KittiObject]) -> int:
    duplicate_like = 0
    for pred in unmatched_preds:
        threshold = DEFAULT_IOU_THRESHOLDS.get(pred.class_name, 0.5)
        best_iou = 0.0
        for gt in gt_objects:
            if gt.class_name != pred.class_name:
                continue
            best_iou = max(best_iou, bev_iou(gt, pred))
        if best_iou >= threshold:
            duplicate_like += 1
    return duplicate_like


def audit_failure_matcher(label_dir: Path, pred_dir: Path, thresholds: Iterable[float]) -> dict:
    frame_ids = _collect_frame_ids(label_dir, pred_dir)
    dontcare_count = 0
    for label_path in label_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("DontCare"):
                dontcare_count += 1

    threshold_rows = []
    unmatched_gt_examples = []
    unmatched_pred_examples = []
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    duplicate_like_fp = 0
    baseline_threshold = min(float(value) for value in thresholds)

    for threshold in thresholds:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        for frame_id in frame_ids:
            gt_objects = read_kitti_objects(label_dir / f"{frame_id}.txt", is_prediction=False)
            pred_objects = [
                obj
                for obj in read_kitti_objects(pred_dir / f"{frame_id}.txt", is_prediction=True)
                if obj.score is None or float(obj.score) >= float(threshold)
            ]
            matches, unmatched_preds, unmatched_gt = greedy_match_objects(gt_objects, pred_objects)
            totals["tp"] += len(matches)
            totals["fp"] += len(unmatched_preds)
            totals["fn"] += len(unmatched_gt)
            if float(threshold) == baseline_threshold:
                duplicate_like_fp += _duplicate_like_fp_count(gt_objects, unmatched_preds)
                for match in matches:
                    by_class[match["gt"].class_name]["tp"] += 1
                for pred in unmatched_preds:
                    by_class[pred.class_name]["fp"] += 1
                    if len(unmatched_pred_examples) < 10 and (pred.score or 0.0) >= 0.5:
                        unmatched_pred_examples.append(
                            {
                                "frame_id": pred.frame_id,
                                "class_name": pred.class_name,
                                "score": pred.score,
                                "distance_m": pred.distance_m,
                            }
                        )
                for gt in unmatched_gt:
                    by_class[gt.class_name]["fn"] += 1
                    if len(unmatched_gt_examples) < 10:
                        unmatched_gt_examples.append(
                            {
                                "frame_id": gt.frame_id,
                                "class_name": gt.class_name,
                                "difficulty": "unknown",
                                "distance_m": gt.distance_m,
                            }
                        )
        threshold_rows.append({"score_threshold": float(threshold), **totals})

    pred_summary = summarize_prediction_dir(pred_dir)
    total_preds = pred_summary["total_box_count"]
    audit_summary = {
        "status": "completed",
        "prediction_dir": str(pred_dir),
        "label_dir": str(label_dir),
        "frame_count": len(frame_ids),
        "pred_summary": pred_summary,
        "tp_fp_fn_by_score_threshold": threshold_rows,
        "tp_fp_fn_by_class": [{"bucket": cls, **counts} for cls, counts in sorted(by_class.items())],
        "duplicate_prediction_rate": (duplicate_like_fp / total_preds) if total_preds > 0 else None,
        "duplicate_like_fp_count": duplicate_like_fp,
        "unsupported_gt_filtering": {
            "dontcare_line_count": dontcare_count,
            "note": "The analysis matcher ignores unsupported classes such as DontCare and does not claim equivalence to the official evaluator.",
        },
        "matcher_checks": {
            "class_aware_matching": True,
            "single_match_per_gt": True,
            "dontcare_explicit_filter": False,
            "difficulty_filter_equivalence": False,
            "iou_metric": "BEV IoU",
            "thresholds": dict(DEFAULT_IOU_THRESHOLDS),
        },
        "unmatched_gt_examples": unmatched_gt_examples,
        "unmatched_high_score_prediction_examples": unmatched_pred_examples,
    }
    return audit_summary


def load_existing_debug_inputs(report_dir: Path) -> dict:
    return {
        "baseline_eval": read_json_or_default(report_dir / "kitti_official_eval.json", {}),
        "wrapper_eval": read_json_or_default(report_dir / "wrapper_pytorch_core_eval.json", {}),
        "bucket_report": read_json_or_default(report_dir / "tensorrt_bucketed_core_report.json", {}),
        "bucket_eval": read_json_or_default(report_dir / "tensorrt_bucketed_kitti_eval.json", {}),
        "raw_diff": read_json_or_default(report_dir / "tensorrt_same_frame_diff.json", {}),
        "failure_summary": read_json_or_default(report_dir / "failure_matcher_summary.json", {}),
    }

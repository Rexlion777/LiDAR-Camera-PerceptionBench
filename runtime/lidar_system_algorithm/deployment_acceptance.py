from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .failure_matcher import (
    SUPPORTED_CLASSES,
    DEFAULT_IOU_THRESHOLDS,
    distance_bin,
    read_kitti_objects,
)
from .report_schema import ensure_dir


RANGE_BINS = ("0-20m", "20-40m", "40-60m", "60m+")
CLASS_NAMES = tuple(SUPPORTED_CLASSES)
SCORE_BINS = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)


@dataclass
class FailureAggregate:
    by_range: list[dict]
    by_class: list[dict]
    totals: dict[str, int]


def deterministic_rng(seed: int, frame_id: str, perturbation_type: str, perturbation_value: str) -> np.random.Generator:
    mix = f"{seed}:{frame_id}:{perturbation_type}:{perturbation_value}"
    state = abs(hash(mix)) % (2**32)
    return np.random.default_rng(state)


def apply_point_perturbation(
    points: np.ndarray,
    perturbation_type: str,
    perturbation_value: str,
    seed: int,
    frame_id: str,
) -> tuple[np.ndarray, dict]:
    stats = {
        "perturbation_type": perturbation_type,
        "perturbation_value": perturbation_value,
        "input_point_count": int(points.shape[0]),
        "output_point_count": int(points.shape[0]),
        "skipped": False,
        "skipped_reason": None,
    }
    if perturbation_type == "baseline":
        return points, stats

    if perturbation_type == "point_dropout":
        ratio = float(perturbation_value)
        rng = deterministic_rng(seed, frame_id, perturbation_type, perturbation_value)
        keep_mask = rng.random(points.shape[0]) >= ratio
        if not np.any(keep_mask):
            keep_mask[rng.integers(0, max(points.shape[0], 1)) % max(points.shape[0], 1)] = True
        out = points[keep_mask]
        stats["output_point_count"] = int(out.shape[0])
        return out, stats

    if perturbation_type == "range_crop":
        if str(perturbation_value).lower() == "full":
            return points, stats
        max_range = float(perturbation_value)
        radial = np.linalg.norm(points[:, :3], axis=1)
        keep_mask = radial <= max_range
        out = points[keep_mask]
        if out.shape[0] == 0:
            stats["skipped"] = True
            stats["skipped_reason"] = f"range_crop={max_range} removed all points"
            return points[:1].copy(), stats
        stats["output_point_count"] = int(out.shape[0])
        return out, stats

    stats["skipped"] = True
    stats["skipped_reason"] = f"Unsupported perturbation for detector input: {perturbation_type}"
    return points, stats


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def prediction_subset_stats(pred_dir: Path, sample_ids: list[str]) -> dict:
    per_frame_rows: list[dict] = []
    score_values: list[float] = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    range_counts = {name: 0 for name in RANGE_BINS}
    total_boxes = 0
    invalid_geometry_count = 0
    empty_prediction_file_count = 0

    for sample_id in sample_ids:
        path = pred_dir / f"{sample_id}.txt"
        objects = read_kitti_objects(path, is_prediction=True)
        if not path.exists() or len(objects) == 0:
            empty_prediction_file_count += 1
        scores = [obj.score for obj in objects if obj.score is not None]
        frame_class_counts = {name: 0 for name in CLASS_NAMES}
        frame_range_counts = {name: 0 for name in RANGE_BINS}
        for obj in objects:
            total_boxes += 1
            frame_class_counts[obj.class_name] = frame_class_counts.get(obj.class_name, 0) + 1
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
            bucket = distance_bin(obj.distance_m)
            frame_range_counts[bucket] = frame_range_counts.get(bucket, 0) + 1
            range_counts[bucket] = range_counts.get(bucket, 0) + 1
            score = _safe_float(obj.score)
            if score is not None:
                score_values.append(score)
            dims = [obj.height, obj.width, obj.length, *obj.location_camera_xyz, obj.rotation_y]
            if obj.height <= 0 or obj.width <= 0 or obj.length <= 0 or not all(np.isfinite(dims)):
                invalid_geometry_count += 1
        per_frame_rows.append(
            {
                "frame_id": sample_id,
                "predicted_box_count": int(len(objects)),
                "car_count": int(frame_class_counts.get("Car", 0)),
                "ped_count": int(frame_class_counts.get("Pedestrian", 0)),
                "cyc_count": int(frame_class_counts.get("Cyclist", 0)),
                "score_mean": float(np.mean(scores)) if scores else None,
                "score_p50": float(np.percentile(scores, 50.0)) if scores else None,
                "score_p95": float(np.percentile(scores, 95.0)) if scores else None,
                "range_0_20": int(frame_range_counts.get("0-20m", 0)),
                "range_20_40": int(frame_range_counts.get("20-40m", 0)),
                "range_40_60": int(frame_range_counts.get("40-60m", 0)),
                "range_60_plus": int(frame_range_counts.get("60m+", 0)),
            }
        )

    score_array = np.asarray(score_values, dtype=np.float64) if score_values else np.zeros((0,), dtype=np.float64)
    score_hist, _ = np.histogram(score_array, bins=SCORE_BINS) if score_array.size else (np.zeros(len(SCORE_BINS) - 1), SCORE_BINS)
    class_hist = np.asarray([class_counts[name] for name in CLASS_NAMES], dtype=np.float64)
    range_hist = np.asarray([range_counts[name] for name in RANGE_BINS], dtype=np.float64)
    return {
        "prediction_file_count": len(sample_ids),
        "empty_prediction_file_count": empty_prediction_file_count,
        "invalid_geometry_count": invalid_geometry_count,
        "total_box_count": total_boxes,
        "per_class_box_count": class_counts,
        "score_summary": {
            "mean": float(score_array.mean()) if score_array.size else None,
            "p50": float(np.percentile(score_array, 50.0)) if score_array.size else None,
            "p95": float(np.percentile(score_array, 95.0)) if score_array.size else None,
            "min": float(score_array.min()) if score_array.size else None,
            "max": float(score_array.max()) if score_array.size else None,
        },
        "histograms": {
            "score": score_hist.tolist(),
            "class": class_hist.tolist(),
            "range": range_hist.tolist(),
        },
        "per_frame_rows": per_frame_rows,
    }


def jensen_shannon_divergence(lhs: list[float] | np.ndarray, rhs: list[float] | np.ndarray) -> float:
    p = np.asarray(lhs, dtype=np.float64)
    q = np.asarray(rhs, dtype=np.float64)
    if p.sum() <= 0 and q.sum() <= 0:
        return 0.0
    if p.sum() <= 0:
        p = np.ones_like(q)
    if q.sum() <= 0:
        q = np.ones_like(p)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def compute_prediction_health(setting_stats: dict, baseline_stats: dict) -> dict:
    base_total = max(float(baseline_stats.get("total_box_count", 0)), 1.0)
    total = float(setting_stats.get("total_box_count", 0))
    prediction_count_drift = abs(total - base_total) / base_total

    score_distribution_drift = jensen_shannon_divergence(
        setting_stats.get("histograms", {}).get("score", []),
        baseline_stats.get("histograms", {}).get("score", []),
    )
    class_distribution_drift = jensen_shannon_divergence(
        setting_stats.get("histograms", {}).get("class", []),
        baseline_stats.get("histograms", {}).get("class", []),
    )
    range_distribution_drift = jensen_shannon_divergence(
        setting_stats.get("histograms", {}).get("range", []),
        baseline_stats.get("histograms", {}).get("range", []),
    )
    denom = max(float(setting_stats.get("prediction_file_count", 0)), 1.0)
    invalid_geometry_rate = float(setting_stats.get("invalid_geometry_count", 0)) / max(total, 1.0)
    empty_prediction_rate = float(setting_stats.get("empty_prediction_file_count", 0)) / denom
    return {
        "prediction_count_drift": prediction_count_drift,
        "score_distribution_drift": score_distribution_drift,
        "class_distribution_drift": class_distribution_drift,
        "range_distribution_drift": range_distribution_drift,
        "invalid_geometry_rate": invalid_geometry_rate,
        "empty_prediction_rate": empty_prediction_rate,
    }


def aggregate_failure_metrics(label_dir: Path, pred_dir: Path, sample_ids: list[str]) -> FailureAggregate:
    by_range = {bucket: {"tp": 0, "fp": 0, "fn": 0} for bucket in RANGE_BINS}
    by_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    total_tp = total_fp = total_fn = 0

    for sample_id in sample_ids:
        gt_objects = read_kitti_objects(label_dir / f"{sample_id}.txt", is_prediction=False)
        pred_objects = read_kitti_objects(pred_dir / f"{sample_id}.txt", is_prediction=True)
        remaining_gt = {index: obj for index, obj in enumerate(gt_objects)}
        sorted_preds = sorted(pred_objects, key=lambda obj: float(obj.score or 0.0), reverse=True)
        for pred in sorted_preds:
            best_idx = None
            best_iou = -1.0
            for gt_index, gt in remaining_gt.items():
                if gt.class_name != pred.class_name:
                    continue
                threshold = DEFAULT_IOU_THRESHOLDS.get(pred.class_name, 0.5)
                iou = bev_iou_fast(gt, pred)
                if iou >= threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = gt_index
            if best_idx is None:
                total_fp += 1
                by_class[pred.class_name]["fp"] += 1
                by_range[distance_bin(pred.distance_m)]["fp"] += 1
                continue
            gt = remaining_gt.pop(best_idx)
            total_tp += 1
            by_class[gt.class_name]["tp"] += 1
            by_range[distance_bin(gt.distance_m)]["tp"] += 1
        for gt in remaining_gt.values():
            total_fn += 1
            by_class[gt.class_name]["fn"] += 1
            by_range[distance_bin(gt.distance_m)]["fn"] += 1

    def _rows(table: dict[str, dict[str, int]], name_key: str) -> list[dict]:
        rows = []
        for bucket, counts in table.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            rows.append({name_key: bucket, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall})
        return rows

    return FailureAggregate(
        by_range=_rows(by_range, "range_bin"),
        by_class=_rows(by_class, "class_name"),
        totals={"tp": total_tp, "fp": total_fp, "fn": total_fn},
    )


def bev_iou_fast(lhs, rhs) -> float:
    import cv2

    lhs_poly = bev_polygon_camera_fast(lhs).astype(np.float32)
    rhs_poly = bev_polygon_camera_fast(rhs).astype(np.float32)
    inter_area, _ = cv2.intersectConvexConvex(lhs_poly, rhs_poly)
    if inter_area <= 0.0:
        return 0.0
    lhs_area = abs(cv2.contourArea(lhs_poly))
    rhs_area = abs(cv2.contourArea(rhs_poly))
    union = lhs_area + rhs_area - float(inter_area)
    return float(inter_area / union) if union > 0 else 0.0


def bev_polygon_camera_fast(obj) -> np.ndarray:
    x, _, z = obj.location_camera_xyz
    cx = float(z)
    cy = float(x)
    dx = float(obj.length)
    dy = float(obj.width)
    heading = float(obj.rotation_y)
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    local = np.array(
        [
            [dx / 2, dy / 2],
            [dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2, dy / 2],
        ],
        dtype=np.float64,
    )
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    return (local @ rot.T) + np.array([[cx, cy]], dtype=np.float64)


def filter_prediction_dir_by_score(src_dir: Path, dst_dir: Path, sample_ids: list[str], threshold: float) -> None:
    ensure_dir(dst_dir)
    for sample_id in sample_ids:
        src = src_dir / f"{sample_id}.txt"
        dst = dst_dir / f"{sample_id}.txt"
        if not src.exists():
            dst.write_text("", encoding="utf-8")
            continue
        kept: list[str] = []
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 16:
                continue
            try:
                score = float(parts[15])
            except Exception:
                continue
            if score >= threshold:
                kept.append(line)
        dst.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def copy_csv_to_origin(src_csv: Path, origin_dir: Path) -> Path:
    ensure_dir(origin_dir)
    dst = origin_dir / src_csv.name
    shutil.copyfile(src_csv, dst)
    return dst


def write_origin_friendly_csv(path: Path, rows: list[dict], fieldnames: list[str], origin_dir: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    copy_csv_to_origin(path, origin_dir)


def write_setting_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

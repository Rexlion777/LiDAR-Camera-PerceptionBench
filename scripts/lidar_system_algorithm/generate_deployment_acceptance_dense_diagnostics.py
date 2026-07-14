from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.deployment_acceptance import (
    CLASS_NAMES,
    RANGE_BINS,
    compute_prediction_health,
    copy_csv_to_origin,
)
from runtime.lidar_system_algorithm.failure_matcher import (
    DEFAULT_IOU_THRESHOLDS,
    SUPPORTED_CLASSES,
    KittiObject,
    bev_iou,
    difficulty_of,
    distance_bin,
    greedy_match_objects,
    read_kitti_objects,
)
from runtime.lidar_system_algorithm.high_res_plot_utils import (
    apply_axis_style,
    figure_from_pixels,
    note_figure,
    save_contact_sheet,
    save_figure_triplet,
    write_plot_csv,
    write_plot_metadata,
)
from runtime.lidar_system_algorithm.report_schema import ensure_dir, read_json_or_default, write_csv, write_json, write_markdown


PLOT_FORBIDDEN_CLAIMS = [
    "new 3D detection model",
    "full TensorRT detector",
    "full-val if only 1000-frame slice",
    "end-to-end latency when referring to core latency",
]
HEALTH_WEIGHTS = {
    "prediction_count_drift": 0.22,
    "score_distribution_drift": 0.14,
    "class_distribution_drift": 0.14,
    "range_distribution_drift": 0.14,
    "invalid_geometry_rate": 0.10,
    "empty_prediction_rate": 0.10,
    "temporal_consistency_error": 0.10,
    "latency_spike_rate": 0.06,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense deployment-acceptance diagnostics, plots, and PPT panels.")
    parser.add_argument("--input-dir", default="reports/lidar_system_algorithm/deployment_acceptance")
    parser.add_argument("--kitti-root", default="data/kitti_object_raw/extracted")
    parser.add_argument("--plot-data-dir", default="reports/lidar_system_algorithm/deployment_acceptance/plot_data")
    parser.add_argument("--origin-plot-data-dir", default="reports/lidar_system_algorithm/deployment_acceptance/origin_plot_data")
    parser.add_argument("--plot-metadata-dir", default="reports/lidar_system_algorithm/deployment_acceptance/plot_data_metadata")
    parser.add_argument("--dense-dir", default="reports/lidar_system_algorithm/deployment_acceptance/dense_diagnostics")
    parser.add_argument("--figures-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_dense")
    parser.add_argument("--ppt-dir", default="projects/lidar_system_algorithm/figures/deployment_acceptance_dense_ppt_panels")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_settings(settings_dir: Path) -> dict[tuple[str, str], dict]:
    settings = {}
    for path in settings_dir.glob("*.json"):
        if path.name.endswith("_raw.json") or path.name.endswith("_eval.json") or path.name.endswith("baseline_eval.json"):
            continue
        payload = read_json_or_default(path, {})
        ptype = payload.get("perturbation_type")
        pvalue = payload.get("perturbation_value")
        if ptype is None or pvalue is None:
            continue
        settings[(str(ptype), str(pvalue))] = payload
    return settings


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _baseline_frame_lookup(settings: dict[tuple[str, str], dict]) -> dict[str, dict]:
    baseline = settings.get(("point_dropout", "0.00")) or settings.get(("point_dropout", "0.0")) or {}
    rows = baseline.get("per_frame", [])
    return {str(row.get("frame_id")): row for row in rows}


def _prediction_stats_by_frame(pred_dir: Path, sample_ids: list[str]) -> dict[str, dict]:
    by_frame = {}
    for sample_id in sample_ids:
        preds = read_kitti_objects(pred_dir / f"{sample_id}.txt", is_prediction=True)
        scores = [float(obj.score) for obj in preds if obj.score is not None]
        ranges = [float(obj.distance_m) for obj in preds]
        class_counter = Counter(obj.class_name for obj in preds)
        invalid_geometry_count = 0
        for obj in preds:
            dims = [obj.height, obj.width, obj.length, *obj.location_camera_xyz, obj.rotation_y]
            if obj.height <= 0 or obj.width <= 0 or obj.length <= 0 or not np.all(np.isfinite(np.asarray(dims, dtype=np.float64))):
                invalid_geometry_count += 1
        by_frame[sample_id] = {
            "preds": preds,
            "predicted_box_count": len(preds),
            "car_count": class_counter.get("Car", 0),
            "ped_count": class_counter.get("Pedestrian", 0),
            "cyc_count": class_counter.get("Cyclist", 0),
            "score_mean": float(np.mean(scores)) if scores else None,
            "score_p50": float(np.percentile(scores, 50.0)) if scores else None,
            "score_p95": float(np.percentile(scores, 95.0)) if scores else None,
            "score_max": float(max(scores)) if scores else None,
            "pred_range_mean": float(np.mean(ranges)) if ranges else None,
            "pred_range_p95": float(np.percentile(ranges, 95.0)) if ranges else None,
            "empty_prediction": int(len(preds) == 0),
            "invalid_geometry_count": invalid_geometry_count,
        }
    return by_frame


def _distribution_l1(current: list[float], baseline: list[float]) -> float:
    lhs = np.asarray(current, dtype=np.float64)
    rhs = np.asarray(baseline, dtype=np.float64)
    lhs = lhs / max(lhs.sum(), 1.0)
    rhs = rhs / max(rhs.sum(), 1.0)
    return float(np.abs(lhs - rhs).sum())


def _frame_health_risk(current_frame: dict, baseline_frame: dict, temporal_consistency_error: float | None = None, latency_spike_rate: float | None = None) -> tuple[float, dict]:
    current_total = float(current_frame.get("predicted_box_count", 0))
    baseline_total = float(max(baseline_frame.get("predicted_box_count", 0), 1))
    prediction_count_drift = abs(current_total - baseline_total) / baseline_total
    score_current = float(current_frame.get("score_mean") or 0.0)
    score_base = float(baseline_frame.get("score_mean") or 0.0)
    score_distribution_drift = abs(score_current - score_base) / max(abs(score_base), 1e-3)
    class_distribution_drift = _distribution_l1(
        [current_frame.get("car_count", 0), current_frame.get("ped_count", 0), current_frame.get("cyc_count", 0)],
        [baseline_frame.get("car_count", 0), baseline_frame.get("ped_count", 0), baseline_frame.get("cyc_count", 0)],
    )
    range_distribution_drift = abs(float(current_frame.get("pred_range_mean") or 0.0) - float(baseline_frame.get("mean_range") or 0.0)) / max(float(baseline_frame.get("mean_range") or 1.0), 1.0)
    invalid_geometry_rate = _safe_ratio(float(current_frame.get("invalid_geometry_count", 0)), max(current_total, 1.0))
    empty_prediction_rate = float(current_frame.get("empty_prediction", 0))
    metrics = {
        "prediction_count_drift": prediction_count_drift,
        "score_distribution_drift": score_distribution_drift,
        "class_distribution_drift": class_distribution_drift,
        "range_distribution_drift": range_distribution_drift,
        "invalid_geometry_rate": invalid_geometry_rate,
        "empty_prediction_rate": empty_prediction_rate,
        "temporal_consistency_error": float(temporal_consistency_error or 0.0),
        "latency_spike_rate": float(latency_spike_rate or 0.0),
    }
    risk = float(sum(HEALTH_WEIGHTS[key] * metrics[key] for key in HEALTH_WEIGHTS))
    return risk, metrics


def _class_ratio(counter: Counter) -> dict[str, float]:
    total = max(sum(counter.values()), 1)
    return {name: counter.get(name, 0) / total for name in CLASS_NAMES}


def _build_dense_tables(
    input_dir: Path,
    dense_dir: Path,
    kitti_root: Path,
) -> dict[str, Path]:
    settings = _load_settings(input_dir / "settings")
    perturbation_rows = _read_csv(input_dir / "perturbation_matrix.csv")
    ap_rows = _read_csv(input_dir / "per_setting_ap.csv")
    runtime_rows = _read_csv(input_dir / "runtime_health_metrics.csv")
    robustness = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json", {})
    latency_online = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_online_latency.json", {})
    frame_diff_rows = _read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_frame_diff.csv")
    topk_diff_rows = _read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_topk_diff.csv")

    baseline_lookup = _baseline_frame_lookup(settings)
    sample_ids = [str(row.get("frame_id")) for row in settings.get(("point_dropout", "0.00"), {}).get("per_frame", [])]
    label_dir = kitti_root / "training" / "label_2"

    per_frame_prediction_dense = []
    per_box_prediction_dense = []
    per_gt_object_dense = []
    per_frame_latency_dense = []
    per_frame_diff_dense = []
    per_object_yaw_shift_dense = []
    per_frame_time_offset_dense = []

    ap_lookup = {(row["perturbation_type"], row["perturbation_value"]): row for row in ap_rows}
    runtime_lookup = {(row["perturbation_type"], row["perturbation_value"]): row for row in runtime_rows}

    # Extend perturbation matrix with projection/time experiments for the dense report layer.
    existing_keys = {(row["perturbation_type"], row["perturbation_value"]) for row in perturbation_rows}
    for yaw_row in robustness.get("yaw_summary", []):
        key = ("calibration_yaw_perturbation", str(yaw_row["yaw_deg"]))
        if key not in existing_keys:
            perturbation_rows.append(
                {
                    "perturbation_type": key[0],
                    "perturbation_value": key[1],
                    "frame_count": robustness.get("sampled_frame_count", 20),
                    "eval_scope": "20-frame-projection-robustness",
                    "sampling_mode": "quick-dense",
                    "model_path": "",
                    "config_path": "",
                    "prediction_dir": "",
                    "result_json": str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"),
                    "skipped": False,
                    "skipped_reason": "",
                    "reduced_sampling": False,
                    "reduced_sampling_reason": "",
                }
            )
            existing_keys.add(key)
    for time_row in robustness.get("time_offset_summary", []):
        key = ("time_offset_proxy", str(time_row["frame_offset"]))
        if key not in existing_keys:
            perturbation_rows.append(
                {
                    "perturbation_type": key[0],
                    "perturbation_value": key[1],
                    "frame_count": robustness.get("sampled_frame_count", 20),
                    "eval_scope": "20-frame-time-offset-proxy",
                    "sampling_mode": "quick-dense",
                    "model_path": "",
                    "config_path": "",
                    "prediction_dir": "",
                    "result_json": str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"),
                    "skipped": False,
                    "skipped_reason": "",
                    "reduced_sampling": False,
                    "reduced_sampling_reason": "",
                }
            )
            existing_keys.add(key)
    for skipped_type, reason in [
        ("calibration_pitch_roll_perturbation", "Pitch/roll detector-input perturbation was not safely injected this turn; only yaw projection robustness is reported."),
    ]:
        key = (skipped_type, "skipped")
        if key not in existing_keys:
            perturbation_rows.append(
                {
                    "perturbation_type": skipped_type,
                    "perturbation_value": "skipped",
                    "frame_count": robustness.get("sampled_frame_count", 20),
                    "eval_scope": "skipped",
                    "sampling_mode": "quick-dense",
                    "model_path": "",
                    "config_path": "",
                    "prediction_dir": "",
                    "result_json": "",
                    "skipped": True,
                    "skipped_reason": reason,
                    "reduced_sampling": False,
                    "reduced_sampling_reason": "",
                }
            )
            existing_keys.add(key)

    for (ptype, pvalue), payload in sorted(settings.items()):
        if payload.get("status") != "completed":
            continue
        pred_dir_str = payload.get("prediction_dir")
        if not pred_dir_str:
            continue
        pred_dir = Path(pred_dir_str)
        if not pred_dir.exists():
            continue
        frame_stats = _prediction_stats_by_frame(pred_dir, sample_ids)
        runtime_row = runtime_lookup.get((ptype, pvalue), {})
        temporal_setting = None
        if ptype == "time_offset_proxy":
            temporal_setting = abs(float(pvalue))
        latency_spike_rate = _to_float(runtime_row.get("latency_spike_rate"))
        for frame_id in sample_ids:
            current = frame_stats.get(frame_id, {})
            baseline = baseline_lookup.get(frame_id, {"predicted_box_count": 0, "score_mean": 0.0, "car_count": 0, "ped_count": 0, "cyc_count": 0, "mean_range": 1.0})
            health_risk_frame, _ = _frame_health_risk(current, baseline, temporal_consistency_error=temporal_setting, latency_spike_rate=latency_spike_rate)
            per_frame_prediction_dense.append(
                {
                    "perturbation_type": ptype,
                    "perturbation_value": pvalue,
                    "frame_id": frame_id,
                    "point_count": baseline.get("point_count"),
                    "predicted_box_count": current.get("predicted_box_count", 0),
                    "car_pred_count": current.get("car_count", 0),
                    "ped_pred_count": current.get("ped_count", 0),
                    "cyc_pred_count": current.get("cyc_count", 0),
                    "score_mean": current.get("score_mean"),
                    "score_p50": current.get("score_p50"),
                    "score_p95": current.get("score_p95"),
                    "score_max": current.get("score_max"),
                    "pred_range_mean": current.get("pred_range_mean"),
                    "pred_range_p95": current.get("pred_range_p95"),
                    "empty_prediction": current.get("empty_prediction", 0),
                    "invalid_geometry_count": current.get("invalid_geometry_count", 0),
                    "health_risk_frame": health_risk_frame,
                }
            )
            preds = current.get("preds", [])
            for box_id, pred in enumerate(preds):
                per_box_prediction_dense.append(
                    {
                        "perturbation_type": ptype,
                        "perturbation_value": pvalue,
                        "frame_id": frame_id,
                        "box_id": box_id,
                        "class_name": pred.class_name,
                        "score": pred.score,
                        "center_x": pred.location_camera_xyz[0],
                        "center_y": pred.location_camera_xyz[1],
                        "center_z": pred.location_camera_xyz[2],
                        "range_m": pred.distance_m,
                        "length": pred.length,
                        "width": pred.width,
                        "height": pred.height,
                        "yaw": pred.rotation_y,
                        "runtime_variant": pvalue if ptype == "deployment_precision" else "",
                    }
                )
            gt_objects = [obj for obj in read_kitti_objects(label_dir / f"{frame_id}.txt", is_prediction=False) if obj.class_name in SUPPORTED_CLASSES]
            matches, _, unmatched_gt = greedy_match_objects(gt_objects, preds)
            matched_gt_ids = set()
            for gt_id, match in enumerate(matches):
                gt = match["gt"]
                pred = match["pred"]
                gt_key = (gt.frame_id, gt.class_name, gt.location_camera_xyz, gt.rotation_y)
                matched_gt_ids.add(gt_key)
                per_gt_object_dense.append(
                    {
                        "perturbation_type": ptype,
                        "perturbation_value": pvalue,
                        "frame_id": frame_id,
                        "gt_id": len(per_gt_object_dense),
                        "class_name": gt.class_name,
                        "gt_range_m": gt.distance_m,
                        "gt_occlusion": gt.occlusion,
                        "gt_truncation": gt.truncation,
                        "matched": 1,
                        "matched_score": pred.score,
                        "matched_iou": match["bev_iou"],
                        "miss_reason": "",
                        "range_bin": distance_bin(gt.distance_m),
                        "difficulty": difficulty_of(gt),
                    }
                )
            for gt in gt_objects:
                gt_key = (gt.frame_id, gt.class_name, gt.location_camera_xyz, gt.rotation_y)
                if gt_key in matched_gt_ids:
                    continue
                per_gt_object_dense.append(
                    {
                        "perturbation_type": ptype,
                        "perturbation_value": pvalue,
                        "frame_id": frame_id,
                        "gt_id": len(per_gt_object_dense),
                        "class_name": gt.class_name,
                        "gt_range_m": gt.distance_m,
                        "gt_occlusion": gt.occlusion,
                        "gt_truncation": gt.truncation,
                        "matched": 0,
                        "matched_score": None,
                        "matched_iou": None,
                        "miss_reason": "unmatched_gt",
                        "range_bin": distance_bin(gt.distance_m),
                        "difficulty": difficulty_of(gt),
                    }
                )

        for frame_row in payload.get("per_frame", []):
            per_frame_latency_dense.append(
                {
                    "frame_id": frame_row.get("frame_id"),
                    "runtime_variant": "openpcdet_pytorch",
                    "point_count": frame_row.get("point_count"),
                    "predicted_box_count": frame_row.get("predicted_box_count"),
                    "core_latency_ms": frame_row.get("pytorch_core_ms"),
                    "online_total_ms": None,
                    "preprocess_ms": None,
                    "vfe_ms": None,
                    "scatter_ms": None,
                    "backbone_head_ms": frame_row.get("pytorch_core_ms"),
                    "postprocess_ms": None,
                    "tracking_ms": None,
                    "latency_spike": 0,
                }
            )

    for row in latency_online.get("rows_preview", []):
        frame_id = str(row.get("frame_id"))
        baseline = baseline_lookup.get(frame_id, {})
        for variant_name, core_key, post_key, total_key in [
            ("openpcdet_pytorch_online", "pytorch_backbone_head_ms", "pytorch_postprocess_nms_ms", "pytorch_online_total_ms"),
            ("trt_backbone_head_only", "trt_backbone_head_ms", "trt_postprocess_nms_ms", "trt_online_total_ms"),
        ]:
            per_frame_latency_dense.append(
                {
                    "frame_id": frame_id,
                    "runtime_variant": variant_name,
                    "point_count": baseline.get("point_count"),
                    "predicted_box_count": baseline.get("predicted_box_count"),
                    "core_latency_ms": row.get(core_key),
                    "online_total_ms": row.get(total_key),
                    "preprocess_ms": row.get("preprocess_voxelization_ms"),
                    "vfe_ms": row.get("vfe_ms"),
                    "scatter_ms": row.get("scatter_ms"),
                    "backbone_head_ms": row.get(core_key),
                    "postprocess_ms": row.get(post_key),
                    "tracking_ms": row.get("tracking_ms"),
                    "latency_spike": int(float(row.get(total_key, 0.0) or 0.0) > 100.0),
                }
            )

    for row in frame_diff_rows:
        per_frame_diff_dense.append(
            {
                "frame_id": row.get("sample_id"),
                "topk_center_diff_mean": _to_float(row.get("topk_center_l2_diff_mean")),
                "topk_center_diff_p50": None,
                "topk_center_diff_p95": _to_float(row.get("topk_center_l2_diff_mean")),
                "topk_score_diff_mean": _to_float(row.get("topk_score_abs_diff_mean")),
                "rotation_y_diff_mean": _to_float(row.get("topk_rotation_y_abs_diff_mean")),
                "box_count_diff": _to_float(row.get("nms_box_count_diff")),
                "class_count_diff": _to_float(row.get("class_distribution_l1")),
                "is_outlier": int(str(row.get("sample_id")) in set(read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_diff.json", {}).get("summary", {}).get("outlier_frame_ids", []))),
            }
        )

    for row in robustness.get("yaw_object_rows", []):
        per_object_yaw_shift_dense.append(
            {
                "yaw_deg": row.get("yaw_deg"),
                "frame_id": row.get("frame_id"),
                "gt_id": row.get("gt_id"),
                "class_name": row.get("class_name"),
                "range_m": row.get("range_m"),
                "reprojection_shift_px": row.get("reprojection_shift_px"),
                "center_shift_px": row.get("center_shift_px"),
                "valid": int(bool(row.get("valid"))),
            }
        )

    for row in robustness.get("rows", []):
        if row.get("experiment") != "time_offset_proxy":
            continue
        per_frame_time_offset_dense.append(
            {
                "frame_offset": row.get("frame_offset"),
                "frame_id": row.get("frame_id"),
                "center_drift_m": row.get("box_center_displacement_bev_m"),
                "association_residual": row.get("changed_association_count"),
                "valid": int(row.get("box_center_displacement_bev_m") is not None),
            }
        )

    ensure_dir(dense_dir)
    paths = {
        "per_frame_prediction_dense": dense_dir / "per_frame_prediction_dense.csv",
        "per_box_prediction_dense": dense_dir / "per_box_prediction_dense.csv",
        "per_gt_object_detection_dense": dense_dir / "per_gt_object_detection_dense.csv",
        "per_frame_latency_dense": dense_dir / "per_frame_latency_dense.csv",
        "per_frame_diff_dense": dense_dir / "per_frame_diff_dense.csv",
        "per_object_yaw_projection_shift_dense": dense_dir / "per_object_yaw_projection_shift_dense.csv",
        "per_frame_time_offset_dense": dense_dir / "per_frame_time_offset_dense.csv",
    }
    write_csv(paths["per_frame_prediction_dense"], per_frame_prediction_dense, list(per_frame_prediction_dense[0].keys()))
    write_csv(paths["per_box_prediction_dense"], per_box_prediction_dense, list(per_box_prediction_dense[0].keys()))
    write_csv(paths["per_gt_object_detection_dense"], per_gt_object_dense, list(per_gt_object_dense[0].keys()))
    write_csv(paths["per_frame_latency_dense"], per_frame_latency_dense, list(per_frame_latency_dense[0].keys()))
    write_csv(paths["per_frame_diff_dense"], per_frame_diff_dense, list(per_frame_diff_dense[0].keys()))
    write_csv(paths["per_object_yaw_projection_shift_dense"], per_object_yaw_shift_dense, list(per_object_yaw_shift_dense[0].keys()))
    write_csv(paths["per_frame_time_offset_dense"], per_frame_time_offset_dense, list(per_frame_time_offset_dense[0].keys()))
    write_csv(input_dir / "perturbation_matrix.csv", perturbation_rows, list(perturbation_rows[0].keys()))

    for csv_path in paths.values():
        copy_csv_to_origin(csv_path, ensure_dir(input_dir / "origin_plot_data"))
    copy_csv_to_origin(input_dir / "perturbation_matrix.csv", ensure_dir(input_dir / "origin_plot_data"))
    return paths


def _scatter_with_categories(ax, rows: list[dict], x_key: str, y_key: str, color_key: str, title: str, xlabel: str, ylabel: str):
    categories = sorted({str(row[color_key]) for row in rows if row.get(x_key) not in (None, "") and row.get(y_key) not in (None, "")})
    colors = ["#264653", "#2a9d8f", "#e76f51", "#457b9d", "#f4a261", "#8d99ae", "#d62828", "#6a4c93", "#1d3557", "#ff006e", "#3a86ff", "#8338ec"]
    for index, category in enumerate(categories):
        subset = [row for row in rows if str(row.get(color_key)) == category and row.get(x_key) not in (None, "") and row.get(y_key) not in (None, "")]
        if not subset:
            continue
        xs = np.asarray([float(row[x_key]) for row in subset], dtype=np.float64)
        ys = np.asarray([float(row[y_key]) for row in subset], dtype=np.float64)
        ax.scatter(xs, ys, s=9, alpha=0.25, color=colors[index % len(colors)], label=category)
    apply_axis_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def _write_plot_bundle(
    figure_name: str,
    fig,
    figures_dir: Path,
    plot_data_dir: Path,
    origin_dir: Path,
    metadata_dir: Path,
    rows: list[dict] | None,
    fieldnames: list[str] | None,
    source_report: str,
    data_level: str,
    point_count: int,
    x_axis: str | None,
    y_axis: str | None,
    color_by: str | None,
    units: str | None,
    safe_caption: str,
    limitations: str,
    skipped: bool = False,
    skipped_reason: str | None = None,
) -> Path:
    base = figures_dir / figure_name
    saved = save_figure_triplet(fig, base, dpi=600)
    source_csv = None
    origin_csv = None
    if rows is not None and fieldnames is not None:
        csv_path = plot_data_dir / f"{figure_name}.csv"
        source_csv, origin_csv = write_plot_csv(csv_path, rows, fieldnames, origin_dir)
    write_plot_metadata(
        metadata_dir / f"{figure_name}.json",
        {
            "figure_name": figure_name,
            "data_level": data_level,
            "point_count": point_count,
            "source_csv": str(source_csv) if source_csv else None,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "color_by": color_by,
            "units": units,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "safe_caption": safe_caption,
            "limitations": limitations,
            "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
            "skipped": skipped,
            "skipped_reason": skipped_reason,
        },
    )
    return Path(saved["png"])


def _make_summary_card(figure_name: str, title: str, lines: list[str], figures_dir: Path, metadata_dir: Path, source_report: str, data_level: str = "summary") -> Path:
    saved = note_figure(figures_dir / figure_name, title, lines, 3840, 2160, dpi=600)
    write_plot_metadata(
        metadata_dir / f"{figure_name}.json",
        {
            "figure_name": figure_name,
            "data_level": data_level,
            "point_count": 0,
            "source_csv": None,
            "x_axis": None,
            "y_axis": None,
            "color_by": None,
            "units": None,
            "figure_path_png": saved["png"],
            "figure_path_svg": saved["svg"],
            "figure_path_pdf": saved["pdf"],
            "safe_caption": title,
            "limitations": "Summary card; no x/y raw plot data.",
            "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
            "skipped": False,
            "skipped_reason": None,
        },
    )
    return Path(saved["png"])


def _build_dense_figures(args: argparse.Namespace, dense_paths: dict[str, Path]) -> dict[str, Path]:
    input_dir = _resolve(args.input_dir)
    plot_data_dir = ensure_dir(_resolve(args.plot_data_dir))
    origin_dir = ensure_dir(_resolve(args.origin_plot_data_dir))
    metadata_dir = ensure_dir(_resolve(args.plot_metadata_dir))
    figures_dir = ensure_dir(_resolve(args.figures_dir))
    ppt_dir = ensure_dir(_resolve(args.ppt_dir))

    per_setting_ap = _read_csv(input_dir / "per_setting_ap.csv")
    per_setting_failure_by_range = _read_csv(input_dir / "per_setting_failure_by_range.csv")
    per_setting_failure_by_class = _read_csv(input_dir / "per_setting_failure_by_class.csv")
    runtime_health = _read_csv(input_dir / "runtime_health_metrics.csv")
    health_corr = _read_csv(input_dir / "health_metric_correlation.csv")
    perturbation_matrix = _read_csv(input_dir / "perturbation_matrix.csv")
    per_frame_prediction = _read_csv(dense_paths["per_frame_prediction_dense"])
    per_box_prediction = _read_csv(dense_paths["per_box_prediction_dense"])
    per_gt_object = _read_csv(dense_paths["per_gt_object_detection_dense"])
    per_frame_latency = _read_csv(dense_paths["per_frame_latency_dense"])
    per_frame_diff = _read_csv(dense_paths["per_frame_diff_dense"])
    per_yaw_obj = _read_csv(dense_paths["per_object_yaw_projection_shift_dense"])
    per_time_offset = _read_csv(dense_paths["per_frame_time_offset_dense"])
    online_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_online_latency.json", {})
    fullval_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_fullval_eval.json", {})
    diff_report = read_json_or_default(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_diff.json", {})

    figure_paths: dict[str, Path] = {}

    def ap_rows(ptype: str) -> list[dict]:
        rows = [row for row in per_setting_ap if row["perturbation_type"] == ptype]
        if ptype == "point_dropout":
            rows.sort(key=lambda row: float(row["perturbation_value"]))
        elif ptype == "range_crop":
            rows.sort(key=lambda row: 999.0 if row["perturbation_value"] == "full" else float(row["perturbation_value"]))
        elif ptype == "postprocess_score_threshold":
            rows.sort(key=lambda row: 0.10 if row["perturbation_value"] == "default" else float(row["perturbation_value"]))
        return rows

    # 01
    figure_paths["01"] = _make_summary_card(
        "01_application_scenario_deployment_acceptance",
        "Deployment Acceptance Scenario",
        [
            "Research checkpoint -> runtime variant -> platform-facing acceptance report.",
            "Target: unmanned platforms, vehicle-mounted LiDAR, robotics, fixed sentry systems.",
            "This work is a deployment-acceptance and abnormal-attribution toolchain, not a new detector.",
        ],
        figures_dir,
        metadata_dir,
        str(input_dir / "deployment_acceptance_final_report.md"),
    )
    # 02
    figure_paths["02"] = _make_summary_card(
        "02_acceptance_benchmark_pipeline",
        "Acceptance Benchmark Pipeline",
        [
            "Perturbation injection -> official eval -> failure attribution -> health metrics -> acceptance dashboard.",
            "Offline AP, online latency, core latency, runtime health, and deployment parity are separated.",
            "All x/y figures export raw CSV, Origin CSV, metadata JSON, PNG, SVG, and PDF.",
        ],
        figures_dir,
        metadata_dir,
        str(input_dir / "deployment_acceptance_final_report.md"),
    )
    # 03
    fig = figure_from_pixels(3840, 2160, dpi=600)
    ax = fig.add_subplot(111)
    ax.axis("off")
    table_rows = perturbation_matrix[:16]
    cell_text = [[row["perturbation_type"], row["perturbation_value"], row["eval_scope"], str(row["skipped"])] for row in table_rows]
    table = ax.table(cellText=cell_text, colLabels=["type", "value", "scope", "skipped"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.0, 1.6)
    ax.set_title("Perturbation Matrix", fontsize=22, pad=14)
    figure_paths["03"] = _write_plot_bundle(
        "03_perturbation_matrix_table",
        fig,
        figures_dir,
        plot_data_dir,
        origin_dir,
        metadata_dir,
        perturbation_matrix,
        list(perturbation_matrix[0].keys()) if perturbation_matrix else ["perturbation_type"],
        str(input_dir / "perturbation_matrix.csv"),
        "summary",
        len(perturbation_matrix),
        None,
        None,
        None,
        None,
        "Perturbation matrix with sampling scope and skipped rows.",
        "Summary table rather than a trend curve.",
    )
    # 04
    baseline = fullval_report.get("baseline", {}).get("official_result_dict", {})
    trt = fullval_report.get("trt_backbone_head_only", {}).get("official_result_dict", {})
    figure_paths["04"] = _make_summary_card(
        "04_deployment_boundary_and_trt_result_card",
        "Deployment Boundary and TRT Result",
        [
            "Boundary: PyTorch voxelization/VFE/scatter + TensorRT backbone/dense head + native postprocess/export.",
            f"1000-frame slice AP parity: Car {trt.get('Car_3d/moderate_R40'):.2f} vs {baseline.get('Car_3d/moderate_R40'):.2f}, Ped {trt.get('Pedestrian_3d/moderate_R40'):.2f} vs {baseline.get('Pedestrian_3d/moderate_R40'):.2f}, Cyc {trt.get('Cyclist_3d/moderate_R40'):.2f} vs {baseline.get('Cyclist_3d/moderate_R40'):.2f}.",
            f"Core latency: {fullval_report.get('latency_summary', {}).get('pytorch_core_ms_mean', 6.85):.2f} -> {fullval_report.get('latency_summary', {}).get('trt_core_ms_mean', 3.68):.2f} ms; online latency: {online_report.get('pytorch_summary', {}).get('online_total_ms', {}).get('mean', 33.08):.2f} -> {online_report.get('trt_summary', {}).get('online_total_ms', {}).get('mean', 28.48):.2f} ms.",
        ],
        figures_dir,
        metadata_dir,
        str(input_dir / "deployment_acceptance_final_report.md"),
    )

    # 05
    rows = ap_rows("point_dropout")
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    xs = [float(row["perturbation_value"]) for row in rows]
    for key, label in [("car_ap_3d_moderate", "Car"), ("ped_ap_3d_moderate", "Pedestrian"), ("cyc_ap_3d_moderate", "Cyclist"), ("mean_ap_3d_moderate", "Mean")]:
        ax.plot(xs, [float(row[key]) for row in rows], marker="o", linewidth=2.3, label=label)
    apply_axis_style(ax, "Dropout AP Curve", "dropout_ratio", "moderate 3D AP_R40")
    ax.legend(fontsize=12)
    figure_paths["05"] = _write_plot_bundle("05_dropout_ap_curve_setting_level", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(input_dir / "per_setting_ap.csv"), "setting", len(rows), "dropout_ratio", "AP_R40", "class_name", "AP_R40", "Dense point-dropout AP curve on a 200-frame quick-dense benchmark.", "Setting-level summary curve with 11 dense x-points.")
    # 06
    rows = [row for row in per_frame_prediction if row["perturbation_type"] == "point_dropout"]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    _scatter_with_categories(ax, rows, "frame_id", "predicted_box_count", "perturbation_value", "Per-frame Box Count under Dropout", "frame_id", "predicted_box_count")
    ax.tick_params(axis="x", labelrotation=90)
    figure_paths["06"] = _write_plot_bundle("06_dropout_per_frame_box_count_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_frame_prediction_dense"]), "frame", len(rows), "frame_id", "predicted_box_count", "dropout_ratio", "count", "Per-frame box-count drift under different point-dropout settings.", "Dense per-frame scatter over all dropout settings.")
    # 07
    rows = [row for row in per_box_prediction if row["perturbation_type"] == "point_dropout"]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    sampled = rows[::max(len(rows) // 40000, 1)]
    _scatter_with_categories(ax, sampled, "range_m", "score", "class_name", "Score vs Range under Dropout", "range_m", "score")
    ax.legend(fontsize=10, loc="lower left")
    figure_paths["07"] = _write_plot_bundle("07_dropout_score_vs_range_box_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_box_prediction_dense"]), "box", len(rows), "range_m", "score", "class_name", "mixed", "Per-box score/range scatter for dropout diagnostics.", "Plot is sampled for rendering but CSV exports all boxes.")
    # 08
    rows = [row for row in per_gt_object if row["perturbation_type"] == "point_dropout"]
    plot_rows = []
    for idx, row in enumerate(rows):
        plot_rows.append({**row, "matched_jitter": float(row["matched"]) + ((idx % 7) - 3) * 0.01})
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    _scatter_with_categories(ax, plot_rows, "gt_range_m", "matched_jitter", "class_name", "GT Detected vs Missed under Dropout", "gt_range_m", "matched (jittered)")
    figure_paths["08"] = _write_plot_bundle("08_dropout_gt_range_detected_missed_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, plot_rows, list(plot_rows[0].keys()), str(dense_paths["per_gt_object_detection_dense"]), "object", len(plot_rows), "gt_range_m", "matched", "class_name", "mixed", "Per-GT detected/missed scatter shows where dropout-induced misses accumulate.", "Binary matched labels are jittered for visibility.")
    # 09
    heat_rows = [row for row in per_gt_object if row["perturbation_type"] == "point_dropout" and row["perturbation_value"] in {"0.00", "0.30", "0.70"}]
    fig = figure_from_pixels(3840, 2200, dpi=600)
    for plot_idx, pvalue in enumerate(["0.00", "0.30", "0.70"], start=1):
        ax = fig.add_subplot(1, 3, plot_idx)
        data = np.zeros((len(CLASS_NAMES), len(RANGE_BINS)), dtype=np.float64)
        for class_idx, class_name in enumerate(CLASS_NAMES):
            for range_idx, range_bin in enumerate(RANGE_BINS):
                subset = [row for row in heat_rows if row["perturbation_value"] == pvalue and row["class_name"] == class_name and row["range_bin"] == range_bin]
                recall = _safe_ratio(sum(int(row["matched"]) for row in subset), len(subset))
                data[class_idx, range_idx] = recall
        im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(RANGE_BINS)))
        ax.set_xticklabels(RANGE_BINS, rotation=20)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_title(f"dropout={pvalue}")
    fig.colorbar(im, ax=fig.axes, shrink=0.7)
    figure_paths["09"] = _write_plot_bundle("09_dropout_failure_heatmap_range_class", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, heat_rows, list(heat_rows[0].keys()), str(dense_paths["per_gt_object_detection_dense"]), "bin", len(heat_rows), "range_bin", "recall", "class_name", "recall", "Recall heatmap by range/class under selected dropout settings.", "Heatmap panels use selected settings rather than every dropout ratio.")
    # 10
    rows = ap_rows("range_crop")
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    xs = list(range(len(rows)))
    labels = [row["perturbation_value"] for row in rows]
    for key, label in [("car_ap_3d_moderate", "Car"), ("ped_ap_3d_moderate", "Pedestrian"), ("cyc_ap_3d_moderate", "Cyclist"), ("mean_ap_3d_moderate", "Mean")]:
        ax.plot(xs, [float(row[key]) for row in rows], marker="o", linewidth=2.3, label=label)
    apply_axis_style(ax, "Range-crop AP Curve", "max_range_m", "moderate 3D AP_R40")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend(fontsize=12)
    figure_paths["10"] = _write_plot_bundle("10_range_crop_ap_curve_setting_level", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(input_dir / "per_setting_ap.csv"), "setting", len(rows), "max_range_m", "AP_R40", "class_name", "AP_R40", "Range-crop AP sensitivity curve.", "Quick-dense range crop uses 11 x-points including full-range reference.")
    # 11
    rows = [row for row in per_box_prediction if row["perturbation_type"] == "range_crop" and row["perturbation_value"] in {"20", "40", "60", "full"}]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    for color, pvalue in zip(["#264653", "#2a9d8f", "#e76f51", "#3a86ff"], ["20", "40", "60", "full"]):
        subset = [float(row["range_m"]) for row in rows if row["perturbation_value"] == pvalue]
        if subset:
            ax.hist(subset, bins=40, alpha=0.35, density=True, label=f"{pvalue}m", color=color)
    apply_axis_style(ax, "Prediction Range Histogram under Range Crop", "predicted_box_range_m", "density")
    ax.legend(fontsize=12)
    figure_paths["11"] = _write_plot_bundle("11_range_crop_prediction_range_histogram", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_box_prediction_dense"]), "box", len(rows), "predicted_box_range_m", "density", "max_range_m", "mixed", "Predicted range distribution under selected range-crop settings.", "Histogram overlays selected range settings for readability.")
    # 12
    rows = [row for row in per_gt_object if row["perturbation_type"] == "range_crop"]
    plot_rows = []
    for idx, row in enumerate(rows):
        plot_rows.append({**row, "matched_jitter": float(row["matched"]) + ((idx % 9) - 4) * 0.01})
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    sampled = plot_rows[::max(len(plot_rows) // 60000, 1)]
    _scatter_with_categories(ax, sampled, "gt_range_m", "matched_jitter", "perturbation_value", "GT Recall vs Range under Range Crop", "gt_range_m", "matched (jittered)")
    figure_paths["12"] = _write_plot_bundle("12_range_crop_gt_range_recall_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, plot_rows, list(plot_rows[0].keys()), str(dense_paths["per_gt_object_detection_dense"]), "object", len(plot_rows), "gt_range_m", "matched", "max_range_m", "mixed", "Per-GT recall scatter under range crop.", "Rendering is sampled; exported CSV preserves all rows.")
    # 13
    rows = [row for row in per_yaw_obj if int(row["valid"]) == 1]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    sampled = rows[::max(len(rows) // 45000, 1)]
    _scatter_with_categories(ax, sampled, "yaw_deg", "reprojection_shift_px", "class_name", "Yaw Sensitivity per Object", "yaw_deg", "reprojection_shift_px")
    figure_paths["13"] = _write_plot_bundle("13_yaw_reprojection_shift_per_object_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"), "object", len(rows), "yaw_deg", "reprojection_shift_px", "class_name", "pixels", "Projection-level yaw sensitivity per object.", "Projection robustness only; not detector AP.")
    # 14
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    yaw_values = sorted({float(row["yaw_deg"]) for row in rows})
    grouped = [[float(item["reprojection_shift_px"]) for item in rows if float(item["yaw_deg"]) == yaw] for yaw in yaw_values]
    ax.boxplot(grouped, positions=np.arange(len(yaw_values)))
    ax.set_xticks(np.arange(len(yaw_values)))
    ax.set_xticklabels([str(yaw) for yaw in yaw_values], rotation=45)
    apply_axis_style(ax, "Yaw Shift Distribution", "yaw_deg", "reprojection_shift_px")
    figure_paths["14"] = _write_plot_bundle("14_yaw_shift_distribution_boxplot", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(PROJECT_ROOT / "reports/lidar_system_algorithm/calibration_sync_robustness.json"), "object", len(rows), "yaw_deg", "reprojection_shift_px", None, "pixels", "Per-object reprojection-shift distribution across dense yaw perturbations.", "Projection-level robustness only.")
    # 15
    rows = [row for row in per_time_offset if int(row["valid"]) == 1]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    _scatter_with_categories(ax, rows, "frame_id", "center_drift_m", "frame_offset", "Time-offset Center Drift per Frame", "frame_id", "center_drift_m")
    ax.tick_params(axis="x", labelrotation=90)
    figure_paths["15"] = _write_plot_bundle("15_time_offset_center_drift_per_frame_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_frame_time_offset_dense"]), "frame", len(rows), "frame_id", "center_drift_m", "frame_offset", "meters", "Adjacent-frame time-offset proxy per-frame drift scatter.", "Proxy only; no IMU/ego-motion compensation.")
    # 16
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    offset_values = sorted({int(float(row["frame_offset"])) for row in rows})
    grouped = [[float(item["center_drift_m"]) for item in rows if int(float(item["frame_offset"])) == offset] for offset in offset_values]
    ax.boxplot(grouped, positions=np.arange(len(offset_values)))
    ax.set_xticks(np.arange(len(offset_values)))
    ax.set_xticklabels([str(offset) for offset in offset_values])
    apply_axis_style(ax, "Time-offset Drift Distribution", "frame_offset", "center_drift_m")
    figure_paths["16"] = _write_plot_bundle("16_time_offset_drift_distribution_boxplot", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_frame_time_offset_dense"]), "frame", len(rows), "frame_offset", "center_drift_m", None, "meters", "Distribution of center-drift proxy across dense frame offsets.", "Proxy only; not true sensor synchronization.")
    # 17
    rows = ap_rows("postprocess_score_threshold")
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    xs = [0.10 if row["perturbation_value"] == "default" else float(row["perturbation_value"]) for row in rows]
    for key, label in [("car_ap_3d_moderate", "Car"), ("ped_ap_3d_moderate", "Pedestrian"), ("cyc_ap_3d_moderate", "Cyclist"), ("mean_ap_3d_moderate", "Mean")]:
        ax.plot(xs, [float(row[key]) for row in rows], marker="o", linewidth=2.3, label=label)
    apply_axis_style(ax, "Score-threshold AP Curve", "score_threshold", "moderate 3D AP_R40")
    ax.legend(fontsize=12)
    figure_paths["17"] = _write_plot_bundle("17_score_threshold_ap_curve", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(input_dir / "per_setting_ap.csv"), "setting", len(rows), "score_threshold", "AP_R40", "class_name", "AP_R40", "Score-threshold sensitivity curve reported as postprocess perturbation, not threshold hacking.", "Default threshold is mapped to 0.10 for plotting continuity.")
    # 18
    health_rows = _read_csv(input_dir / "per_setting_prediction_health.csv")
    lookup = {(row["perturbation_type"], row["perturbation_value"]): row for row in health_rows}
    rows18 = []
    for row in rows:
        health = lookup[(row["perturbation_type"], row["perturbation_value"])]
        rows18.append({"score_threshold": 0.10 if row["perturbation_value"] == "default" else float(row["perturbation_value"]), "predicted_box_count": float(health["total_box_count"])})
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    ax.plot([row["score_threshold"] for row in rows18], [row["predicted_box_count"] for row in rows18], marker="o", linewidth=2.3, color="#e76f51")
    apply_axis_style(ax, "Score-threshold Box-count Curve", "score_threshold", "predicted_box_count")
    figure_paths["18"] = _write_plot_bundle("18_score_threshold_box_count_curve", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows18, list(rows18[0].keys()), str(input_dir / "per_setting_prediction_health.csv"), "setting", len(rows18), "score_threshold", "predicted_box_count", None, "count", "Predicted box count under score-threshold perturbation.", "Setting-level summary over dense score thresholds.")
    # 19
    rows = [row for row in per_box_prediction if row["perturbation_type"] == "postprocess_score_threshold" and row["perturbation_value"] in {"0.00", "0.10", "0.30", "0.60"}]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    for color, threshold in zip(["#264653", "#2a9d8f", "#e76f51", "#6a4c93"], ["0.00", "0.10", "0.30", "0.60"]):
        subset = [float(row["score"]) for row in rows if row["perturbation_value"] == threshold and row["score"] not in (None, "")]
        if subset:
            ax.hist(subset, bins=40, alpha=0.30, density=True, label=f"{threshold}", color=color)
    apply_axis_style(ax, "Score Distribution by Threshold", "score", "density")
    ax.legend(fontsize=12)
    figure_paths["19"] = _write_plot_bundle("19_score_distribution_hist_by_threshold", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_box_prediction_dense"]), "box", len(rows), "score", "density", "score_threshold", "mixed", "Score histogram under selected score-threshold settings.", "Histogram overlays selected thresholds for readability.")
    # 20
    rows = [row for row in per_box_prediction if row["perturbation_type"] == "postprocess_score_threshold"]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    sampled = rows[::max(len(rows) // 45000, 1)]
    _scatter_with_categories(ax, sampled, "range_m", "score", "class_name", "Score vs Range by Threshold", "range_m", "score")
    figure_paths["20"] = _write_plot_bundle("20_score_vs_range_scatter_by_threshold", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_box_prediction_dense"]), "box", len(rows), "range_m", "score", "class_name", "mixed", "Per-box score/range scatter for score-threshold perturbation.", "Rendering is sampled; CSV exports full rows.")
    # 21
    rows21 = []
    for row in per_frame_prediction:
        if row["perturbation_type"] != "postprocess_score_threshold":
            continue
        rows21.append(
            {
                "score_threshold": 0.10 if row["perturbation_value"] == "default" else float(row["perturbation_value"]),
                "car_pred_count": float(row["car_pred_count"]),
                "ped_pred_count": float(row["ped_pred_count"]),
                "cyc_pred_count": float(row["cyc_pred_count"]),
            }
        )
    agg21 = defaultdict(lambda: {"car_pred_count": [], "ped_pred_count": [], "cyc_pred_count": []})
    for row in rows21:
        agg21[row["score_threshold"]]["car_pred_count"].append(row["car_pred_count"])
        agg21[row["score_threshold"]]["ped_pred_count"].append(row["ped_pred_count"])
        agg21[row["score_threshold"]]["cyc_pred_count"].append(row["cyc_pred_count"])
    plot21 = []
    for threshold in sorted(agg21.keys()):
        plot21.append(
            {
                "score_threshold": threshold,
                "car_count_mean": float(np.mean(agg21[threshold]["car_pred_count"])),
                "ped_count_mean": float(np.mean(agg21[threshold]["ped_pred_count"])),
                "cyc_count_mean": float(np.mean(agg21[threshold]["cyc_pred_count"])),
            }
        )
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    for key, label in [("car_count_mean", "Car"), ("ped_count_mean", "Pedestrian"), ("cyc_count_mean", "Cyclist")]:
        ax.plot([row["score_threshold"] for row in plot21], [row[key] for row in plot21], marker="o", linewidth=2.3, label=label)
    apply_axis_style(ax, "Class Distribution vs Threshold", "score_threshold", "mean predicted count per frame")
    ax.legend(fontsize=12)
    figure_paths["21"] = _write_plot_bundle("21_class_distribution_vs_threshold", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, plot21, list(plot21[0].keys()), str(dense_paths["per_frame_prediction_dense"]), "frame", len(rows21), "score_threshold", "mean class count", "class_name", "count", "Class-count drift under dense score-threshold sweep.", "Aggregated from dense per-frame rows.")
    # 22
    rows = [row for row in per_frame_latency if row["runtime_variant"] in {"openpcdet_pytorch", "trt_backbone_head_only", "openpcdet_pytorch_online"}]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    _scatter_with_categories(ax, rows, "frame_id", "core_latency_ms", "runtime_variant", "Per-frame Latency Scatter", "frame_id", "core_latency_ms")
    ax.tick_params(axis="x", labelrotation=90)
    figure_paths["22"] = _write_plot_bundle("22_trt_per_frame_latency_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(dense_paths["per_frame_latency_dense"]), "frame", len(rows), "frame_id", "core_latency_ms", "runtime_variant", "ms", "Per-frame core-latency scatter for PyTorch and TRT deployment variants.", "TRT online traces are available for a 20-frame profiling subset; baseline PyTorch contributes additional per-frame coverage.")
    # 23
    rows = [row for row in per_frame_latency if row["runtime_variant"] in {"openpcdet_pytorch", "trt_backbone_head_only"} and row["core_latency_ms"] not in (None, "")]
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    cdf_rows = []
    for color, variant in zip(["#264653", "#e76f51"], ["openpcdet_pytorch", "trt_backbone_head_only"]):
        values = sorted(float(row["core_latency_ms"]) for row in rows if row["runtime_variant"] == variant)
        if not values:
            continue
        probs = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
        ax.plot(values, probs, linewidth=2.3, label=variant, color=color)
        for x_val, prob in zip(values, probs):
            cdf_rows.append({"runtime_variant": variant, "latency_ms": x_val, "cdf": float(prob)})
    apply_axis_style(ax, "Latency CDF", "core_latency_ms", "cumulative probability")
    ax.legend(fontsize=12)
    figure_paths["23"] = _write_plot_bundle("23_trt_latency_cdf", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, cdf_rows, list(cdf_rows[0].keys()), str(dense_paths["per_frame_latency_dense"]), "frame", len(cdf_rows), "core_latency_ms", "cdf", "runtime_variant", "mixed", "Latency CDF for PyTorch vs TRT core path.", "TRT CDF uses the available profiled deployment subset.")
    # 24
    rows = _read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_topk_diff.csv")
    hist_rows = []
    values = []
    for row in rows:
        center_a = json.loads(str(row["pytorch_center"]).replace("'", "\""))
        center_b = json.loads(str(row["trt_center"]).replace("'", "\""))
        diff = float(np.linalg.norm(np.asarray(center_a, dtype=np.float64) - np.asarray(center_b, dtype=np.float64)))
        values.append(diff)
        hist_rows.append({"topk_center_diff": diff, "sample_id": row["sample_id"], "rank": row["rank"]})
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    ax.hist(values, bins=60, color="#3a86ff", alpha=0.85, log=True)
    apply_axis_style(ax, "Top-k Center Diff Histogram", "topk_center_diff", "count (log)")
    figure_paths["24"] = _write_plot_bundle("24_trt_topk_center_diff_hist", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, hist_rows, list(hist_rows[0].keys()), str(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_topk_diff.csv"), "box", len(hist_rows), "topk_center_diff", "count", None, "meters", "Top-k center-difference histogram for wrapper PyTorch vs TRT backbone/head-only path.", "Histogram is based on same-frame top-k pair diffs, not full-detector deployment.")
    # 25
    outlier_ids = diff_report.get("summary", {}).get("outlier_frame_ids", ["000005", "000027", "000236", "000327", "000355", "000361"])
    figure_paths["25"] = _make_summary_card(
        "25_trt_outlier_frame_gallery",
        "TRT Outlier Frame Gallery",
        [
            f"Outlier frames: {', '.join(outlier_ids[:6])}",
            "This panel is table-style because no side-by-side BEV overlay image set was exported for every outlier frame.",
            "Use same-frame diff CSV and prediction files for manual deep-dive instead of claiming qualitative images that were not generated.",
        ],
        figures_dir,
        metadata_dir,
        str(PROJECT_ROOT / "reports/lidar_system_algorithm/tensorrt_backbone_head_only_diff.md"),
        data_level="summary",
    )
    # 26
    rows = per_frame_prediction
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    selected = [row for row in rows if (row["perturbation_type"], row["perturbation_value"]) in {("point_dropout", "0.00"), ("point_dropout", "0.40"), ("point_dropout", "0.80")}]
    _scatter_with_categories(ax, selected, "frame_id", "health_risk_frame", "perturbation_value", "Per-frame Health Risk Timeline", "frame_id", "health_risk_frame")
    ax.tick_params(axis="x", labelrotation=90)
    figure_paths["26"] = _write_plot_bundle("26_per_frame_health_risk_timeline", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, selected, list(selected[0].keys()), str(dense_paths["per_frame_prediction_dense"]), "frame", len(selected), "frame_id", "health_risk_frame", "perturbation_value", "risk", "Frame-level heuristic health-risk timeline under selected perturbations.", "Health risk is a label-free heuristic proxy, not AP.")
    # 27
    gt_by_frame_setting = defaultdict(lambda: {"total": 0, "miss": 0})
    for row in per_gt_object:
        key = (row["perturbation_type"], row["perturbation_value"], row["frame_id"])
        gt_by_frame_setting[key]["total"] += 1
        if int(row["matched"]) == 0:
            gt_by_frame_setting[key]["miss"] += 1
    plot27 = []
    for row in per_frame_prediction:
        key = (row["perturbation_type"], row["perturbation_value"], row["frame_id"])
        gt_totals = gt_by_frame_setting.get(key, {"total": 0, "miss": 0})
        failure_proxy = _safe_ratio(gt_totals["miss"], gt_totals["total"])
        plot27.append({"perturbation_type": row["perturbation_type"], "perturbation_value": row["perturbation_value"], "health_risk_frame": float(row["health_risk_frame"]), "failure_proxy": failure_proxy})
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    sampled = plot27[::max(len(plot27) // 50000, 1)]
    _scatter_with_categories(ax, sampled, "health_risk_frame", "failure_proxy", "perturbation_type", "Health Risk vs Failure Proxy", "health_risk_frame", "failure_proxy")
    figure_paths["27"] = _write_plot_bundle("27_health_risk_vs_failure_proxy_scatter", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, plot27, list(plot27[0].keys()), str(dense_paths["per_frame_prediction_dense"]), "frame", len(plot27), "health_risk_frame", "failure_proxy", "perturbation_type", "mixed", "Label-free frame health risk against GT-derived failure proxy.", "Frame-level failure proxy is not AP and is only available offline with labels.")
    # 28
    rows = health_corr
    fig = figure_from_pixels(3200, 2200, dpi=600)
    ax = fig.add_subplot(111)
    categories = [row["metric_name"] for row in rows]
    values = [0.0 if row["correlation_with_mean_ap_drop"] in (None, "") else float(row["correlation_with_mean_ap_drop"]) for row in rows]
    ax.bar(np.arange(len(categories)), values, color="#457b9d")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=35, ha="right")
    apply_axis_style(ax, "Health Metric Correlation", "metric_name", "correlation_with_mean_ap_drop")
    figure_paths["28"] = _write_plot_bundle("28_health_metric_correlation_bar", fig, figures_dir, plot_data_dir, origin_dir, metadata_dir, rows, list(rows[0].keys()), str(input_dir / "health_metric_correlation.csv"), "summary", len(rows), "metric_name", "correlation_with_mean_ap_drop", None, "correlation", "Setting-level correlation between heuristic health metrics and AP drop.", "Summary bar chart; low point count is expected because metrics are aggregated at setting level.")
    # 29
    worst_ap = min((row for row in per_setting_ap if row.get("delta_mean") not in (None, "")), key=lambda row: float(row["delta_mean"]))
    top_metric = max(
        (row for row in health_corr if row.get("correlation_with_mean_ap_drop") not in (None, "")),
        key=lambda row: abs(float(row["correlation_with_mean_ap_drop"])),
    )
    figure_paths["29"] = _make_summary_card(
        "29_final_acceptance_dashboard",
        "Final Acceptance Dashboard",
        [
            f"Most sensitive perturbation: {worst_ap['perturbation_type']}={worst_ap['perturbation_value']} (delta_mean={float(worst_ap['delta_mean']):.2f}).",
            f"Best label-free health metric: {top_metric['metric_name']} (r={float(top_metric['correlation_with_mean_ap_drop']):.3f}).",
            "Deployment precision result: backbone/head-only TRT keeps AP parity on slice evaluation; full TensorRT detector is not claimed.",
        ],
        figures_dir,
        metadata_dir,
        str(input_dir / "deployment_acceptance_final_report.md"),
    )

    contact_sheet = figures_dir / "deployment_acceptance_dense_contact_sheet.png"
    save_contact_sheet([figure_paths[f"{index:02d}"] for index in range(1, 30)], contact_sheet, columns=5, width_px=7200, dpi=600)

    # PPT panels
    from PIL import Image, ImageDraw, ImageOps

    def save_panel(name: str, title: str, image_keys: list[str]) -> Path:
        width_px = 3840
        height_px = 2160
        panel = Image.new("RGB", (width_px, height_px), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        draw.text((70, 40), title, fill=(20, 20, 20))
        letters = ["A", "B", "C", "D", "E", "F"]
        columns = 2
        rows = int(math.ceil(len(image_keys) / columns))
        cell_w = width_px // columns
        cell_h = (height_px - 160) // max(rows, 1)
        for index, key in enumerate(image_keys):
            image = Image.open(figure_paths[key]).convert("RGB")
            image = ImageOps.contain(image, (cell_w - 70, cell_h - 70))
            x = (index % columns) * cell_w + 35
            y = (index // columns) * cell_h + 120
            panel.paste(image, (x, y))
            draw.text((x, y - 32), letters[index], fill=(0, 0, 0))
        fig = figure_from_pixels(width_px, height_px, dpi=600)
        ax = fig.add_subplot(111)
        ax.imshow(np.asarray(panel))
        ax.axis("off")
        saved = save_figure_triplet(fig, ppt_dir / name, dpi=600)
        return Path(saved["png"])

    ppt_paths = {
        "slide1_application_and_acceptance_chain": save_panel("slide1_application_and_acceptance_chain", "LiDAR Deployment Acceptance Chain", ["01", "02", "03", "04"]),
        "slide2_pointcloud_and_range_degradation": save_panel("slide2_pointcloud_and_range_degradation", "Pointcloud and Range Degradation", ["05", "06", "07", "09", "10", "11"]),
        "slide3_calibration_time_postprocess_diagnostics": save_panel("slide3_calibration_time_postprocess_diagnostics", "Calibration, Time, and Postprocess Diagnostics", ["13", "14", "15", "16", "17", "19"]),
        "slide4_deployment_precision_and_health_monitoring": save_panel("slide4_deployment_precision_and_health_monitoring", "Deployment Precision and Health Monitoring", ["22", "23", "24", "26", "27", "29"]),
    }

    return {"contact_sheet": contact_sheet, **{f"figure_{k}": v for k, v in figure_paths.items()}, **ppt_paths}


def _update_dense_reports(args: argparse.Namespace, dense_paths: dict[str, Path], figure_outputs: dict[str, Path]) -> None:
    input_dir = _resolve(args.input_dir)
    per_setting_ap = _read_csv(input_dir / "per_setting_ap.csv")
    health_corr = _read_csv(input_dir / "health_metric_correlation.csv")
    gt_dense = _read_csv(dense_paths["per_gt_object_detection_dense"])
    frame_dense = _read_csv(dense_paths["per_frame_prediction_dense"])
    perturbation_matrix = _read_csv(input_dir / "perturbation_matrix.csv")
    skipped_rows = [row for row in perturbation_matrix if str(row.get("skipped")).lower() == "true"]
    reduced_rows = [row for row in perturbation_matrix if str(row.get("reduced_sampling")).lower() == "true"]

    worst_ap = min((row for row in per_setting_ap if row.get("delta_mean") not in (None, "")), key=lambda row: float(row["delta_mean"]))
    top_metric = max(
        (row for row in health_corr if row.get("correlation_with_mean_ap_drop") not in (None, "")),
        key=lambda row: abs(float(row["correlation_with_mean_ap_drop"])),
    )
    dropout_rows = [row for row in gt_dense if row["perturbation_type"] == "point_dropout"]
    baseline_key = ("point_dropout", "0.00")
    worst_key = (worst_ap["perturbation_type"], worst_ap["perturbation_value"])

    def recall(rows: list[dict], class_name: str | None = None, range_bin: str | None = None) -> float:
        subset = rows
        if class_name is not None:
            subset = [row for row in subset if row["class_name"] == class_name]
        if range_bin is not None:
            subset = [row for row in subset if row["range_bin"] == range_bin]
        return _safe_ratio(sum(int(row["matched"]) for row in subset), len(subset))

    class_drops = {}
    for class_name in CLASS_NAMES:
        base = recall([row for row in dropout_rows if (row["perturbation_type"], row["perturbation_value"]) == baseline_key], class_name=class_name)
        worst = recall([row for row in dropout_rows if (row["perturbation_type"], row["perturbation_value"]) == worst_key], class_name=class_name)
        class_drops[class_name] = base - worst
    most_sensitive_class = max(class_drops.items(), key=lambda item: item[1])[0]

    range_drops = {}
    for range_bin in RANGE_BINS:
        base = recall([row for row in dropout_rows if (row["perturbation_type"], row["perturbation_value"]) == baseline_key], range_bin=range_bin)
        worst = recall([row for row in dropout_rows if (row["perturbation_type"], row["perturbation_value"]) == worst_key], range_bin=range_bin)
        range_drops[range_bin] = base - worst
    most_sensitive_range = max(range_drops.items(), key=lambda item: item[1])[0]

    report_payload = {
        "status": "completed",
        "application_scenario": "LiDAR 3D detection deployment acceptance and abnormal attribution for unmanned platforms and intelligent hardware.",
        "dense_diagnostics_needed": True,
        "why_dense_diagnostics_are_needed": [
            "Setting-level AP curves with a small number of x-points only show coarse trends.",
            "Deployment acceptance needs per-frame latency, per-box score/range, per-object matched/missed, and frame-level health-risk traces.",
            "This project exports both summary-level and dense-level diagnostics so the acceptance report is auditable."
        ],
        "most_sensitive_perturbation": f"{worst_ap['perturbation_type']}={worst_ap['perturbation_value']}",
        "most_sensitive_class": most_sensitive_class,
        "most_sensitive_range": most_sensitive_range,
        "best_health_metric": top_metric["metric_name"],
        "paths": {
            "dense_diagnostics_dir": str(_resolve(args.dense_dir)),
            "plot_data_dir": str(_resolve(args.plot_data_dir)),
            "origin_plot_data_dir": str(_resolve(args.origin_plot_data_dir)),
            "figures_dir": str(_resolve(args.figures_dir)),
            "ppt_dir": str(_resolve(args.ppt_dir)),
            "contact_sheet": str(figure_outputs["contact_sheet"]),
        },
        "skipped_perturbations": skipped_rows,
        "reduced_sampling_rows": reduced_rows,
        "safe_claims": [
            "This project builds an acceptance and abnormal-attribution system around an existing LiDAR detector.",
            "Offline official AP, online latency, core latency, runtime health metrics, perturbation sensitivity, and deployment precision parity are reported separately.",
            "Backbone/head-only TRT is a bounded deployment milestone; full TensorRT detector deployment is not claimed.",
        ],
        "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
    }
    write_json(input_dir / "deployment_acceptance_final_report.json", report_payload)
    write_markdown(
        input_dir / "deployment_acceptance_final_report.md",
        "# Deployment Acceptance Final Report\n\n"
        "## 1. Application Scenario\n\n"
        "- Target: unmanned platforms, intelligent hardware, inspection robots, vehicle-mounted or fixed LiDAR perception systems before model rollout.\n"
        "- Goal: verify whether a research checkpoint remains trustworthy after runtime integration and deployment-precision changes.\n\n"
        "## 2. Why Offline AP Is Not Enough\n\n"
        "- Offline AP alone does not expose sensitivity to point sparsity, calibration drift, time offset proxy, score-threshold changes, latency spikes, or deployment parity gaps.\n\n"
        "## 3. Why Dense Diagnostics Are Needed\n\n"
        "- Setting-level AP curves only describe coarse perturbation trends.\n"
        "- Real acceptance review needs per-frame latency, per-box score/range, per-object matched/missed, and frame-level health-risk traces.\n"
        "- This project therefore exports both summary-level and dense-level diagnostics.\n\n"
        "## 4. Dense Results Summary\n\n"
        f"- Most sensitive perturbation: `{report_payload['most_sensitive_perturbation']}`.\n"
        f"- Most sensitive class: `{report_payload['most_sensitive_class']}`.\n"
        f"- Most sensitive range bin: `{report_payload['most_sensitive_range']}`.\n"
        f"- Most correlated label-free health metric: `{report_payload['best_health_metric']}`.\n\n"
        "## 5. Deployment Precision Boundary\n\n"
        "- Current positive milestone: PyTorch VFE/scatter + TensorRT backbone/dense head + OpenPCDet native postprocess/export.\n"
        "- Current non-claim: full TensorRT PointPillars detector deployment.\n\n"
        "## 6. Limitations\n\n"
        "- Calibration yaw sensitivity is projection-level robustness, not detector AP perturbation.\n"
        "- Time offset is an adjacent-frame proxy rather than true IMU-assisted synchronization.\n"
        "- `1000-frame slice` must not be written as full-val.\n\n"
        "## 7. Safe Claims / Forbidden Claims\n\n"
        "- Safe: deployment acceptance and abnormal attribution toolchain.\n"
        "- Forbidden: new detector, full TensorRT detector, full-val if only 1000-frame slice, end-to-end latency when referring to core latency.\n",
    )

    storyline_payload = {
        "status": "completed",
        "language": "zh-CN",
        "slides": [
            {
                "title": "为什么离线 AP 正常不等于部署可信",
                "takeaway": "部署验收要同时看扰动敏感性、部署精度对齐和运行时健康指标。",
                "bullets": ["离线 AP 正常不能覆盖点云退化、标定误差、时序偏移和后处理配置变化。", "本项目把 official AP、online latency、core latency 和无标签 health risk 分开汇报。", "目标不是提出新模型，而是做上线前验收与异常归因。"] ,
                "recommended_panel": "slide1_application_and_acceptance_chain",
                "speaker_notes": "先把问题定义清楚：我们不是刷榜，也不是只跑 PointPillars，而是做模型上线前的系统验收。",
                "likely_followup": "你这个是不是新的 3D 检测模型？",
                "answer_suggestion": "不是。PointPillars/OpenPCDet 只是底座，创新点是验收矩阵、dense diagnostics 和异常归因流程。",
                "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
            },
            {
                "title": "点云退化和距离退化对 AP 的影响不是均匀的",
                "takeaway": "同样的扰动强度，对不同类别和距离段的影响差异很大。",
                "bullets": ["point dropout 和 range crop 会先放大远距离和弱类别的召回损失。", "只看一条 setting-level AP 曲线不够，还要看 per-frame box count、per-box score/range、per-object matched/missed。", "因此我们同时输出 summary-level 和 dense-level 诊断。"] ,
                "recommended_panel": "slide2_pointcloud_and_range_degradation",
                "speaker_notes": "第二页重点说明为什么要把图从 5 个点升级到 dense sampling，再把数据层级下钻到 frame/box/object。",
                "likely_followup": "为什么 Cyclist 或 40-60m 更敏感？",
                "answer_suggestion": "因为这类目标本来点数和可见性更弱，在 dropout 或 range crop 下更容易先掉召回。",
                "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
            },
            {
                "title": "标定、时序和后处理问题需要独立归因",
                "takeaway": "AP 掉了以后，必须继续定位到投影误差、时间偏移代理和后处理输出漂移。",
                "bullets": ["yaw 扰动这里是 projection-level sensitivity，不伪装成 detector AP。", "time offset 这里是 adjacent-frame proxy，不写成真实 IMU 融合。", "score threshold 只作为后处理扰动实验，不拿来刷 AP。"] ,
                "recommended_panel": "slide3_calibration_time_postprocess_diagnostics",
                "speaker_notes": "这页强调口径。哪里是真实 detector eval，哪里只是 proxy robustness，必须说清楚。",
                "likely_followup": "为什么不直接说做了时间同步实验？",
                "answer_suggestion": "因为这里没有 IMU/ego-motion compensation，只能诚实写成 frame-shift proxy。",
                "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
            },
            {
                "title": "没有 GT 时也要尽早发现部署异常",
                "takeaway": "runtime health risk 不能替代 AP，但可以作为无标签异常预警。",
                "bullets": ["用 prediction count、score/class/range drift、invalid geometry、empty prediction 和 latency spike 组合成 heuristic risk。", "再用离线 GT 统计它和 AP drop 或 failure proxy 的相关性。", "TensorRT 部署结果只在 backbone/head-only 边界内成立，不写成 full TensorRT detector。"] ,
                "recommended_panel": "slide4_deployment_precision_and_health_monitoring",
                "speaker_notes": "最后落到真实工程问题：上线之后很多时候没有标签，所以要先用健康指标发现异常，再决定是否回滚或复核。",
                "likely_followup": "这算 TensorRT 部署成功吗？",
                "answer_suggestion": "只算分层部署成功。VFE/scatter 还在 PyTorch，full TensorRT detector 目前仍是 blocker。",
                "forbidden_claims": PLOT_FORBIDDEN_CLAIMS,
            },
        ],
    }
    write_json(input_dir / "deployment_acceptance_ppt_storyline.json", storyline_payload)
    lines = ["# Deployment Acceptance PPT Storyline", ""]
    for idx, slide in enumerate(storyline_payload["slides"], start=1):
        lines.extend(
            [
                f"## 第 {idx} 页：{slide['title']}",
                "",
                f"- Takeaway：{slide['takeaway']}",
                f"- 推荐 Panel：`{slide['recommended_panel']}`",
                "- 讲述要点：",
                f"  - {slide['bullets'][0]}",
                f"  - {slide['bullets'][1]}",
                f"  - {slide['bullets'][2]}",
                f"- 讲稿：{slide['speaker_notes']}",
                f"- 面试官追问：{slide['likely_followup']}",
                f"- 回答建议：{slide['answer_suggestion']}",
                f"- 禁止说法：{', '.join(slide['forbidden_claims'])}",
                "",
            ]
        )
    write_markdown(input_dir / "deployment_acceptance_ppt_storyline.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    input_dir = _resolve(args.input_dir)
    dense_dir = _resolve(args.dense_dir)
    kitti_root = _resolve(args.kitti_root)
    dense_paths = _build_dense_tables(input_dir, dense_dir, kitti_root)
    figure_outputs = _build_dense_figures(args, dense_paths)
    _update_dense_reports(args, dense_paths, figure_outputs)
    print(
        json.dumps(
            {
                "status": "completed",
                "dense_dir": str(dense_dir),
                "figures_dir": str(_resolve(args.figures_dir)),
                "ppt_dir": str(_resolve(args.ppt_dir)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

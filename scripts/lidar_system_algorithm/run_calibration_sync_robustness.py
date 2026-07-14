from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.calibration import Calibration, parse_calibration_file
from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, locate_default_kitti_root, read_image_bgr, read_kitti_bin, resolve_frame_assets
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json, write_markdown
from runtime.lidar_system_algorithm.transforms import compute_distance_and_azimuth, filter_points_in_image, project_lidar_to_image
from runtime.lidar_system_algorithm.visualization import compose_grid, render_projection_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Camera-LiDAR calibration and time-offset robustness analysis.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames.")
    parser.add_argument("--yaw-perturb-deg", default="-2,-1,-0.5,0,0.5,1,2", help="Comma-separated yaw perturbations.")
    parser.add_argument("--translation-perturb-m", default="-0.2,-0.1,0,0.1,0.2", help="Comma-separated translation perturbations.")
    parser.add_argument("--frame-offsets", default="-2,-1,0,1,2", help="Comma-separated frame offsets for time-sync proxy.")
    parser.add_argument("--inference-json", default="reports/lidar_system_algorithm/pointpillars_inference_detections.json", help="Cached detection JSON.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    return parser.parse_args()


def _floats(csv_text: str) -> list[float]:
    return [float(item) for item in csv_text.split(",") if item.strip()]


def _ints(csv_text: str) -> list[int]:
    return [int(item) for item in csv_text.split(",") if item.strip()]


def _perturb_yaw(calib: Calibration, yaw_deg: float) -> Calibration:
    yaw = math.radians(float(yaw_deg))
    rot = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tr = calib.tr_velo_to_cam.copy()
    tr[:, :3] = tr[:, :3] @ rot
    return Calibration(p2=calib.p2.copy(), r0_rect=calib.r0_rect.copy(), tr_velo_to_cam=tr)


def _perturb_translation(calib: Calibration, axis: str, value: float) -> Calibration:
    tr = calib.tr_velo_to_cam.copy()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    tr[axis_index, 3] += float(value)
    return Calibration(p2=calib.p2.copy(), r0_rect=calib.r0_rect.copy(), tr_velo_to_cam=tr)


def _line_plot(rows: list[dict], x_key: str, y_key: str, title: str, x_label: str, y_label: str) -> np.ndarray:
    canvas = np.full((700, 1100, 3), 248, dtype=np.uint8)
    left, top, width, height = 90, 90, 920, 500
    cv2.putText(canvas, title, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 220), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (120, 120, 120), 2)
    if not rows:
        return canvas
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows if row.get(y_key) is not None]
    if not ys:
        return canvas
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-9:
        max_x += 1.0
    if abs(max_y - min_y) < 1e-9:
        max_y += 1.0
    points = []
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key] or 0.0)
        px = left + int((x - min_x) / (max_x - min_x) * width)
        py = top + height - int((y - min_y) / (max_y - min_y) * height)
        points.append((px, py))
        cv2.circle(canvas, (px, py), 5, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(row[x_key]), (px - 18, top + height + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (left + width // 2 - 80, top + height + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (20, top + height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
    return canvas


def _load_detections(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for item in payload.get("detections", []):
        box = item.get("box_3d_lidar", {})
        grouped.setdefault(str(item.get("frame_id")), []).append(
            {
                "object_id": int(item.get("object_id", 0)),
                "class_name": str(item.get("class_name", "unknown")),
                "score": float(item.get("score", 0.0)),
                "center_xyz": np.asarray([float(box.get("x", 0.0)), float(box.get("y", 0.0)), float(box.get("z", 0.0))], dtype=np.float64),
            }
        )
    return grouped


def _center_shift(a: np.ndarray, b: np.ndarray) -> tuple[float | None, int]:
    if a.size == 0 or b.size == 0:
        return None, 0
    diff = a[:, None, :2] - b[None, :, :2]
    dist = np.linalg.norm(diff, axis=2)
    nearest = dist.min(axis=1)
    return float(np.mean(nearest)), int(np.sum(nearest > 2.0))


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    kitti_root = (PROJECT_ROOT / args.kitti_root).resolve() if args.kitti_root and not Path(args.kitti_root).is_absolute() else (Path(args.kitti_root) if args.kitti_root else locate_default_kitti_root())
    frame_ids = list_frame_ids(kitti_root, limit=args.frames)
    yaw_values = _floats(args.yaw_perturb_deg)
    trans_values = _floats(args.translation_perturb_m)
    offsets = _ints(args.frame_offsets)
    detection_by_frame = _load_detections((PROJECT_ROOT / args.inference_json).resolve() if not Path(args.inference_json).is_absolute() else Path(args.inference_json))

    calibration_rows: list[dict] = []
    yaw_object_rows: list[dict] = []
    overlay_images = []
    sampled_frame_count = 0
    for frame_id in frame_ids:
        assets = resolve_frame_assets(kitti_root, frame_id)
        if assets.calib_path is None or assets.image_path is None:
            continue
        calib = parse_calibration_file(assets.calib_path)
        image = read_image_bgr(assets.image_path)
        if image is None:
            continue
        points = read_kitti_bin(assets.lidar_path)[:, :3]
        if points.shape[0] > 6000:
            points = points[:: max(1, points.shape[0] // 6000)]
        _, base_uv, base_valid = project_lidar_to_image(points, calib)
        base_fov = filter_points_in_image(base_uv, base_valid, image.shape)
        if int(base_fov.sum()) == 0:
            continue
        dets = detection_by_frame.get(frame_id, [])
        det_points = np.asarray([det["center_xyz"] for det in dets], dtype=np.float64) if dets else np.zeros((0, 3), dtype=np.float64)
        det_ranges = np.linalg.norm(det_points[:, :2], axis=1) if det_points.size else np.zeros((0,), dtype=np.float64)
        if det_points.size:
            _, det_base_uv, det_base_valid = project_lidar_to_image(det_points, calib)
            det_base_fov = filter_points_in_image(det_base_uv, det_base_valid, image.shape)
        else:
            det_base_uv = np.zeros((0, 2), dtype=np.float64)
            det_base_fov = np.zeros((0,), dtype=bool)
        sampled_frame_count += 1
        for yaw in yaw_values:
            perturbed = _perturb_yaw(calib, yaw)
            _, uv, valid = project_lidar_to_image(points, perturbed)
            fov = filter_points_in_image(uv, valid, image.shape)
            common = base_fov & fov
            shift = np.linalg.norm(uv[common] - base_uv[common], axis=1) if int(common.sum()) else np.asarray([], dtype=np.float64)
            calibration_rows.append(
                {
                    "experiment": "yaw",
                    "frame_id": frame_id,
                    "axis": "yaw",
                    "perturbation": yaw,
                    "projected_valid_count": int(fov.sum()),
                    "avg_reprojection_shift_px": float(np.mean(shift)) if shift.size else None,
                    "median_reprojection_shift_px": float(np.median(shift)) if shift.size else None,
                    "p95_reprojection_shift_px": float(np.percentile(shift, 95)) if shift.size else None,
                }
            )
            if det_points.size:
                _, det_uv, det_valid = project_lidar_to_image(det_points, perturbed)
                det_fov = filter_points_in_image(det_uv, det_valid, image.shape)
                common_det = det_base_fov & det_fov
                for det_index, det in enumerate(dets):
                    valid_flag = bool(common_det[det_index]) if det_index < common_det.shape[0] else False
                    if valid_flag:
                        center_shift = float(np.linalg.norm(det_uv[det_index] - det_base_uv[det_index]))
                    else:
                        center_shift = None
                    yaw_object_rows.append(
                        {
                            "yaw_deg": yaw,
                            "frame_id": frame_id,
                            "gt_id": det.get("object_id"),
                            "class_name": det.get("class_name"),
                            "range_m": float(det_ranges[det_index]) if det_index < det_ranges.shape[0] else None,
                            "reprojection_shift_px": center_shift,
                            "center_shift_px": center_shift,
                            "valid": valid_flag,
                        }
                    )
            if frame_id == frame_ids[0] and yaw in {min(yaw_values), 0.0, max(yaw_values)}:
                overlay_images.append(render_projection_image(image, uv[fov][:1500], np.ones(int(fov.sum()))[:1500], f"yaw {yaw:+.1f} deg"))
        for axis in ["x", "y", "z"]:
            for value in trans_values:
                perturbed = _perturb_translation(calib, axis, value)
                _, uv, valid = project_lidar_to_image(points, perturbed)
                fov = filter_points_in_image(uv, valid, image.shape)
                common = base_fov & fov
                shift = np.linalg.norm(uv[common] - base_uv[common], axis=1) if int(common.sum()) else np.asarray([], dtype=np.float64)
                calibration_rows.append(
                    {
                        "experiment": "translation",
                        "frame_id": frame_id,
                        "axis": axis,
                        "perturbation": value,
                        "projected_valid_count": int(fov.sum()),
                        "avg_reprojection_shift_px": float(np.mean(shift)) if shift.size else None,
                        "median_reprojection_shift_px": float(np.median(shift)) if shift.size else None,
                        "p95_reprojection_shift_px": float(np.percentile(shift, 95)) if shift.size else None,
                    }
                )

    sync_rows: list[dict] = []
    for index, frame_id in enumerate(frame_ids):
        current_list = detection_by_frame.get(frame_id, [])
        current = np.asarray([item["center_xyz"] for item in current_list], dtype=np.float64) if current_list else np.zeros((0, 3), dtype=np.float64)
        for offset in offsets:
            target_index = index + offset
            if target_index < 0 or target_index >= len(frame_ids):
                continue
            shifted_list = detection_by_frame.get(frame_ids[target_index], [])
            shifted = np.asarray([item["center_xyz"] for item in shifted_list], dtype=np.float64) if shifted_list else np.zeros((0, 3), dtype=np.float64)
            mean_shift, changed = _center_shift(current, shifted)
            sync_rows.append(
                {
                    "experiment": "time_offset_proxy",
                    "frame_id": frame_id,
                    "frame_offset": offset,
                    "box_center_displacement_bev_m": mean_shift,
                    "changed_association_count": changed,
                    "current_detection_count": int(current.shape[0]),
                    "shifted_detection_count": int(shifted.shape[0]),
                    "note": "Proxy simulation using adjacent KITTI frames; no IMU/ego-motion compensation is claimed.",
                }
            )

    csv_path = output_dir / "calibration_sync_robustness.csv"
    json_path = output_dir / "calibration_sync_robustness.json"
    md_path = output_dir / "calibration_sync_robustness.md"
    fieldnames = ["experiment", "frame_id", "axis", "perturbation", "frame_offset", "projected_valid_count", "avg_reprojection_shift_px", "median_reprojection_shift_px", "p95_reprojection_shift_px", "box_center_displacement_bev_m", "changed_association_count", "current_detection_count", "shifted_detection_count", "note"]
    rows = calibration_rows + sync_rows
    write_csv(csv_path, rows, fieldnames)
    yaw_summary = []
    for yaw in yaw_values:
        vals = [row["avg_reprojection_shift_px"] for row in calibration_rows if row["experiment"] == "yaw" and row["perturbation"] == yaw and row["avg_reprojection_shift_px"] is not None]
        yaw_summary.append({"yaw_deg": yaw, "avg_reprojection_shift_px": statistics.fmean(vals) if vals else None})
    offset_summary = []
    for offset in offsets:
        vals = [row["box_center_displacement_bev_m"] for row in sync_rows if row["frame_offset"] == offset and row["box_center_displacement_bev_m"] is not None]
        offset_summary.append({"frame_offset": offset, "avg_box_center_displacement_bev_m": statistics.fmean(vals) if vals else None})
    payload = {
        "status": "completed" if rows else "skipped",
        "sampled_frame_count": sampled_frame_count,
        "yaw_summary": yaw_summary,
        "time_offset_summary": offset_summary,
        "yaw_object_rows": yaw_object_rows,
        "limitations": ["Time offset is a proxy based on adjacent KITTI frames; no IMU or ego-motion compensation is used."],
        "rows": rows,
    }
    write_json(json_path, payload)
    write_markdown(
        md_path,
        f"""# Calibration / Time Sync Robustness

- Status: `{payload["status"]}`
- Sampled frames with image+calibration: `{sampled_frame_count}`
- Calibration experiment: yaw and translation perturbations measure image reprojection shift.
- Time sync experiment: adjacent-frame offset is used as a proxy sensitivity analysis, not real IMU fusion.

## Key Result

- yaw +/-2 deg average shift px: `{[row for row in yaw_summary if abs(row['yaw_deg']) == 2.0]}`
- offset +/-2 average BEV displacement m: `{[row for row in offset_summary if abs(row['frame_offset']) == 2]}`
""",
    )
    save_image(figures_dir / "calibration_yaw_reprojection_shift.png", _line_plot(yaw_summary, "yaw_deg", "avg_reprojection_shift_px", "Yaw Perturbation vs Reprojection Shift", "yaw perturbation deg", "avg shift px"))
    save_image(figures_dir / "time_offset_bev_shift.png", _line_plot(offset_summary, "frame_offset", "avg_box_center_displacement_bev_m", "Time Offset Proxy vs BEV Center Shift", "frame offset", "avg BEV shift m"))
    save_image(figures_dir / "calibration_overlay_examples.png", compose_grid(overlay_images[:3], columns=3, label_lines=["Calibration perturbation overlay examples"]))
    print(f"Saved calibration/sync robustness: {json_path}")


if __name__ == "__main__":
    main()

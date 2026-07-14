from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.dbscan_baseline import DetectionBox, build_dbscan_baseline
from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, locate_default_kitti_root, read_kitti_bin, resolve_frame_assets
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json
from runtime.lidar_system_algorithm.tracking import Detection, MultiObjectTracker, OptimizedMultiObjectTracker
from runtime.lidar_system_algorithm.visualization import compose_grid, draw_bar_chart, render_bev, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight LiDAR detection-to-tracking demo.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to process.")
    parser.add_argument("--config", default="projects/lidar_system_algorithm/configs/default.yaml", help="Project config path.")
    parser.add_argument("--inference-json", default="reports/lidar_system_algorithm/pointpillars_inference_detections.json", help="PointPillars detections JSON.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--optimized-association", action="store_true", help="Run vectorized gated association and write optimized tracking artifacts.")
    return parser.parse_args()


def to_detection(frame_id: str, box: DetectionBox) -> Detection:
    return Detection(
        frame_id=frame_id,
        center_xyz=box.center_xyz.astype(float),
        size_xyz=box.size_xyz.astype(float),
        yaw=float(box.yaw),
        score=box.score,
        class_name=box.class_name,
        source=box.source,
    )


def pointpillars_box_to_detection(frame_id: str, item: dict) -> Detection:
    box = item["box_3d_lidar"]
    return Detection(
        frame_id=frame_id,
        center_xyz=np.array([float(box["x"]), float(box["y"]), float(box["z"])], dtype=float),
        size_xyz=np.array([float(box["dx"]), float(box["dy"]), float(box["dz"])], dtype=float),
        yaw=float(box["heading"]),
        score=float(item["score"]) if item.get("score") is not None else None,
        class_name=str(item.get("class_name", "unknown")),
        source="pointpillars",
    )


def main() -> None:
    args = parse_args()
    kitti_root = (
        ((PROJECT_ROOT / args.kitti_root).resolve() if not Path(args.kitti_root).is_absolute() else Path(args.kitti_root).expanduser())
        if args.kitti_root
        else locate_default_kitti_root().resolve()
    )
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    inference_json = (PROJECT_ROOT / args.inference_json).resolve() if not Path(args.inference_json).is_absolute() else Path(args.inference_json)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(((PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)).read_text(encoding="utf-8")) or {}
    dbscan_cfg = config.get("dbscan", {})
    track_cfg = config.get("tracking", {})
    detections_cache = {}
    if inference_json.exists():
        payload = json.loads(inference_json.read_text(encoding="utf-8"))
        for item in payload.get("detections", []):
            detections_cache.setdefault(str(item["frame_id"]), []).append(item)
    if len(detections_cache) >= args.frames:
        frame_ids = sorted(detections_cache.keys())[: args.frames]
    else:
        frame_ids = list_frame_ids(kitti_root, limit=args.frames)
    if not frame_ids:
        raise SystemExit(f"No KITTI frames found under: {kitti_root}")

    legacy_tracker = MultiObjectTracker(
        distance_threshold=float(track_cfg.get("distance_threshold", 4.0)),
        max_age=int(track_cfg.get("max_age", 2)),
        min_hits=int(track_cfg.get("min_hits", 2)),
    )
    tracker = OptimizedMultiObjectTracker(
        distance_threshold=float(track_cfg.get("distance_threshold", 4.0)),
        max_age=int(track_cfg.get("max_age", 2)),
        min_hits=int(track_cfg.get("min_hits", 2)),
    )
    association_latencies: list[float] = []
    legacy_association_latencies: list[float] = []
    per_frame_rows: list[dict] = []
    track_lifetimes: dict[int, list[str]] = {}
    frame_images = []
    used_pointpillars = False

    for frame_id in frame_ids:
        assets = resolve_frame_assets(kitti_root, frame_id)
        points_xyz = read_kitti_bin(assets.lidar_path)[:, :3]
        baseline = None
        if frame_id in detections_cache:
            detections = [pointpillars_box_to_detection(frame_id, item) for item in detections_cache[frame_id]]
            source_mode = "pointpillars"
            used_pointpillars = True
        else:
            baseline = build_dbscan_baseline(
                points_xyz=points_xyz,
                roi=(
                    tuple(dbscan_cfg.get("roi", {}).get("x_range", [0.0, 60.0])),
                    tuple(dbscan_cfg.get("roi", {}).get("y_range", [-25.0, 25.0])),
                    tuple(dbscan_cfg.get("roi", {}).get("z_range", [-2.5, 1.5])),
                ),
                ground_z_threshold=dbscan_cfg.get("ground_z_threshold", -1.35),
                eps=float(dbscan_cfg.get("eps", 0.8)),
                min_points=int(dbscan_cfg.get("min_points", 12)),
                min_cluster_points=int(dbscan_cfg.get("min_cluster_points", 20)),
                sample_step=max(1, int(points_xyz.shape[0] / 12000)),
            )
            detections = [to_detection(frame_id, box) for box in baseline["boxes"]]
            source_mode = "dbscan_baseline_fallback"
        legacy_start = time.perf_counter()
        legacy_tracks = legacy_tracker.update(detections, frame_id=frame_id)
        legacy_latency_ms = (time.perf_counter() - legacy_start) * 1000.0
        legacy_association_latencies.append(legacy_latency_ms)

        start = time.perf_counter()
        tracks, stats = tracker.update_with_stats(detections, frame_id=frame_id)
        association_latencies.append((time.perf_counter() - start) * 1000.0)
        for track in tracks:
            track_lifetimes.setdefault(track.track_id, []).append(frame_id)
        per_frame_rows.append(
            {
                "frame_id": frame_id,
                "detection_count": len(detections),
                "visible_track_count": len(tracks),
                "legacy_visible_track_count": len(legacy_tracks),
                "legacy_association_latency_ms": legacy_latency_ms,
                "association_latency_ms": association_latencies[-1],
                "association_matrix_size": stats["association_matrix_size"],
                "gated_pair_count": stats["gated_pair_count"],
                "spawned_tracks": stats["spawned_tracks"],
                "expired_tracks": stats["expired_tracks"],
                "track_id_switch_proxy": stats["track_id_switch_proxy"],
                "scipy_available": stats["scipy_available"],
                "source_mode": source_mode,
                "dbscan_sample_step": baseline["metadata"]["sample_step"] if baseline is not None else "",
            }
        )
        overlay_boxes = []
        track_labels = {}
        for track in tracks:
            box = DetectionBox(
                cluster_id=track.track_id,
                center_xyz=np.array([track.state[0], track.state[1], 0.0], dtype=float),
                size_xyz=track.size_xyz,
                min_xyz=np.zeros(3, dtype=float),
                max_xyz=np.zeros(3, dtype=float),
                point_count=track.hits,
                distance_m=float(np.linalg.norm(track.state[:2])),
                azimuth_deg=float(np.degrees(np.arctan2(track.state[1], track.state[0]))),
                score=None,
                class_name=track.class_name,
                source="track",
            )
            overlay_boxes.append(box)
            track_labels[track.track_id] = f"track:{track.track_id}"
        frame_images.append(
            render_bev(
                points_xyz if baseline is None else baseline["filtered_points"],
                boxes=overlay_boxes,
                title=f"Tracking {frame_id}",
                extra_lines=[f"detections={len(detections)}", f"tracks={len(tracks)}"],
                track_labels=track_labels,
            )
        )

    track_rows = []
    for track_id, seen_frames in sorted(track_lifetimes.items()):
        track_rows.append(
            {
                "track_id": track_id,
                "first_frame": seen_frames[0],
                "last_frame": seen_frames[-1],
                "lifetime_frames": len(seen_frames),
                "source_mode": "pointpillars" if used_pointpillars else "dbscan_baseline_fallback",
            }
        )

    csv_path = output_dir / "tracking_summary.csv"
    json_path = output_dir / "tracking_summary.json"
    optimized_csv_path = output_dir / "tracking_optimized_summary.csv"
    optimized_json_path = output_dir / "tracking_optimized_summary.json"
    figure_path = figures_dir / "tracking_bev_sequence.png"
    optimized_figure_path = figures_dir / "tracking_bev_sequence_optimized.png"
    latency_figure_path = figures_dir / "tracking_latency_before_after.png"
    write_csv(csv_path, track_rows, ["track_id", "first_frame", "last_frame", "lifetime_frames", "source_mode"])
    write_csv(
        optimized_csv_path,
        per_frame_rows,
        [
            "frame_id",
            "detection_count",
            "visible_track_count",
            "legacy_visible_track_count",
            "legacy_association_latency_ms",
            "association_latency_ms",
            "association_matrix_size",
            "gated_pair_count",
            "spawned_tracks",
            "expired_tracks",
            "track_id_switch_proxy",
            "scipy_available",
            "source_mode",
            "dbscan_sample_step",
        ],
    )
    average_lifetime = statistics.fmean(row["lifetime_frames"] for row in track_rows) if track_rows else 0.0
    scipy_available = bool(per_frame_rows[0]["scipy_available"]) if per_frame_rows else False
    association_method = (
        "vectorized_center_distance_gating_plus_scipy_hungarian"
        if scipy_available
        else "vectorized_center_distance_gating_plus_fallback_assignment"
    )
    association_reason = (
        "Optimized tracker uses vectorized center-distance gating and SciPy Hungarian association on gated cost matrices."
        if scipy_available
        else "Optimized tracker uses vectorized center-distance gating and a local fallback assignment because SciPy is unavailable."
    )
    optimized_payload = {
        "source_mode": "pointpillars" if used_pointpillars else "dbscan_baseline_fallback",
        "association_method": association_method,
        "frame_count": len(frame_ids),
        "average_legacy_association_latency_ms": statistics.fmean(legacy_association_latencies) if legacy_association_latencies else 0.0,
        "average_association_latency_ms": statistics.fmean(association_latencies) if association_latencies else 0.0,
        "max_association_latency_ms": max(association_latencies) if association_latencies else 0.0,
        "average_track_count": statistics.fmean(row["visible_track_count"] for row in per_frame_rows) if per_frame_rows else 0.0,
        "average_track_lifetime": average_lifetime,
        "total_spawned_tracks": sum(int(row["spawned_tracks"]) for row in per_frame_rows),
        "total_expired_tracks": sum(int(row["expired_tracks"]) for row in per_frame_rows),
        "track_id_switch_proxy_total": sum(int(row["track_id_switch_proxy"]) for row in per_frame_rows),
        "scipy_available": scipy_available,
        "target_latency_ms": 5.0,
        "status": "completed" if association_latencies else "skipped",
        "reason": association_reason,
        "per_frame": per_frame_rows,
        "per_track": track_rows,
    }
    write_json(optimized_json_path, optimized_payload)
    write_json(
        json_path,
        {
            "source_mode": "pointpillars" if used_pointpillars else "dbscan_baseline_fallback",
            "reason": "Tracking uses cached PointPillars detections when available; otherwise it falls back to the DBSCAN baseline.",
            "frame_count": len(frame_ids),
            "average_track_count": statistics.fmean(row["visible_track_count"] for row in per_frame_rows) if per_frame_rows else 0.0,
            "average_association_latency_ms": statistics.fmean(association_latencies) if association_latencies else 0.0,
            "average_legacy_association_latency_ms": statistics.fmean(legacy_association_latencies) if legacy_association_latencies else 0.0,
            "max_association_latency_ms": max(association_latencies) if association_latencies else 0.0,
            "dbscan_tracking_note": "DBSCAN in the fallback path uses adaptive point subsampling to keep offline demo latency bounded.",
            "per_frame": per_frame_rows,
            "per_track": track_rows,
        },
    )
    save_image(
        figure_path,
        compose_grid(
            frame_images[: min(6, len(frame_images))],
            columns=3,
            label_lines=["Detection -> Tracking", f"Source: {'PointPillars' if used_pointpillars else 'DBSCAN fallback'}"],
        ),
    )
    save_image(optimized_figure_path, compose_grid(frame_images[: min(6, len(frame_images))], columns=3, label_lines=["Optimized Detection -> Tracking", f"Source: {'PointPillars' if used_pointpillars else 'DBSCAN fallback'}"]))
    save_image(
        latency_figure_path,
        draw_bar_chart(
            [
                {"stage": "legacy_association_ms", "mean_ms": optimized_payload["average_legacy_association_latency_ms"]},
                {"stage": "optimized_association_ms", "mean_ms": optimized_payload["average_association_latency_ms"]},
            ],
            title="Tracking Association Latency Before/After",
        ),
    )
    print(f"Saved tracking summary: {json_path}")
    print(f"Saved optimized tracking summary: {optimized_json_path}")


if __name__ == "__main__":
    main()

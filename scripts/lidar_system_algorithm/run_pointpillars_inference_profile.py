from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.calibration import parse_calibration_file
from runtime.lidar_system_algorithm.dbscan_baseline import DetectionBox, build_dbscan_baseline
from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, locate_default_kitti_root, read_image_bgr, resolve_frame_assets
from runtime.lidar_system_algorithm.openpcdet_adapter import build_openpcdet_env, probe_openpcdet, probe_python_runtime
from runtime.lidar_system_algorithm.online_latency import compute_online_debug_records, summarize_online_latency
from runtime.lidar_system_algorithm.profiling import aggregate_stage_records
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json
from runtime.lidar_system_algorithm.visualization import draw_bar_chart, draw_lidar_boxes_on_image, render_bev, save_image


STAGE_NAMES = [
    "data_load_ms",
    "calibration_parse_ms",
    "point_preprocess_ms",
    "voxelization_or_pillarization_ms",
    "model_forward_ms",
    "backbone_ms",
    "head_ms",
    "nms_ms",
    "postprocess_ms",
    "visualization_ms",
    "total_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the PointPillars/OpenPCDet runtime chain.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--openpcdet-root", default="external/OpenPCDet", help="OpenPCDet root path.")
    parser.add_argument("--cfg-file", default="external/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml", help="PointPillars config path.")
    parser.add_argument("--ckpt", default="checkpoints/pointpillar_kitti.pth", help="PointPillars checkpoint path.")
    parser.add_argument("--python-exe", default="python", help="OpenPCDet runtime Python executable.")
    parser.add_argument("--frames", type=int, default=20, help="Number of measured frames.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    parser.add_argument("--config", default="projects/lidar_system_algorithm/configs/default.yaml", help="Project config path.")
    parser.add_argument("--separate-visualization", action="store_true", help="Write online-vs-debug latency outputs that exclude visualization from online_total_ms.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def to_detection_box(detection: dict) -> DetectionBox:
    box = detection["box_3d_lidar"]
    center_xyz = [float(box["x"]), float(box["y"]), float(box["z"])]
    size_xyz = [float(box["dx"]), float(box["dy"]), float(box["dz"])]
    distance_m = float((center_xyz[0] ** 2 + center_xyz[1] ** 2 + center_xyz[2] ** 2) ** 0.5)
    azimuth_deg = float(math.degrees(math.atan2(center_xyz[1], center_xyz[0])))
    return DetectionBox(
        cluster_id=int(detection.get("object_id", 0)),
        center_xyz=np.array(center_xyz, dtype=float),
        size_xyz=np.array(size_xyz, dtype=float),
        min_xyz=np.array(center_xyz, dtype=float) - np.array(size_xyz, dtype=float) / 2.0,
        max_xyz=np.array(center_xyz, dtype=float) + np.array(size_xyz, dtype=float) / 2.0,
        point_count=0,
        distance_m=distance_m,
        azimuth_deg=azimuth_deg,
        score=float(detection["score"]) if detection.get("score") is not None else None,
        class_name=str(detection.get("class_name", "unknown")),
        yaw=float(box["heading"]),
        source="pointpillars",
    )


def write_helper_input(frames: list[dict]) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="lidar_system_algorithm_"))
    input_path = temp_dir / "input_frames.json"
    input_path.write_text(json.dumps({"frames": frames}, indent=2, ensure_ascii=False), encoding="utf-8")
    return input_path


def call_openpcdet_helper(
    python_exe: Path,
    openpcdet_root: Path,
    cfg_file: Path,
    ckpt_path: Path,
    frames: list[dict],
    warmup: int,
) -> dict:
    helper_script = PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "openpcdet_runtime_helper.py"
    windows_compat_dir = PROJECT_ROOT / "day09_openpcdet" / "windows_compat"
    input_path = write_helper_input(frames)
    output_path = input_path.parent / "openpcdet_runtime_output.json"
    env = build_openpcdet_env(python_executable=python_exe, extra_pythonpaths=[windows_compat_dir, openpcdet_root])
    command = [
        str(python_exe),
        str(helper_script),
        "--openpcdet-root",
        str(openpcdet_root),
        "--cfg-file",
        str(cfg_file),
        "--ckpt",
        str(ckpt_path),
        "--input-json",
        str(input_path),
        "--output-json",
        str(output_path),
        "--warmup",
        str(warmup),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"OpenPCDet helper failed.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    kitti_root = (
        ((PROJECT_ROOT / args.kitti_root).resolve() if not Path(args.kitti_root).is_absolute() else Path(args.kitti_root).expanduser())
        if args.kitti_root
        else locate_default_kitti_root().resolve()
    )
    openpcdet_root = (PROJECT_ROOT / args.openpcdet_root).resolve() if not Path(args.openpcdet_root).is_absolute() else Path(args.openpcdet_root)
    cfg_file = (PROJECT_ROOT / args.cfg_file).resolve() if not Path(args.cfg_file).is_absolute() else Path(args.cfg_file)
    ckpt_path = Path(args.ckpt).expanduser()
    python_exe = Path(args.python_exe).expanduser()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    config = load_yaml((PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_openpcdet(openpcdet_root=openpcdet_root, config_path=cfg_file, checkpoint_path=ckpt_path)
    runtime_probe = (
        probe_python_runtime(python_executable=python_exe, env=build_openpcdet_env(python_executable=python_exe, extra_pythonpaths=[openpcdet_root]))
        if python_exe.exists()
        else {"executable": str(python_exe), "probe_error": "python_executable_missing", "torch": {"cuda_available": False}, "modules": {}}
    )
    if not probe.inference_available or not runtime_probe.get("torch", {}).get("cuda_available", False):
        raise SystemExit("PointPillars runtime is not available. Check OpenPCDet root, checkpoint, and CUDA python environment.")

    total_needed = args.frames + args.warmup
    frame_ids = list_frame_ids(kitti_root, limit=total_needed)
    if len(frame_ids) < total_needed:
        raise SystemExit(f"Need at least {total_needed} frames under {kitti_root}, found {len(frame_ids)}")

    frames = []
    assets_by_frame = {}
    for frame_id in frame_ids:
        assets = resolve_frame_assets(kitti_root, frame_id)
        frames.append({"frame_id": assets.frame_id, "lidar_path": str(assets.lidar_path)})
        assets_by_frame[assets.frame_id] = assets

    helper_payload = call_openpcdet_helper(
        python_exe=python_exe,
        openpcdet_root=openpcdet_root,
        cfg_file=cfg_file,
        ckpt_path=ckpt_path,
        frames=frames,
        warmup=args.warmup,
    )

    detections_by_frame: dict[str, list[dict]] = {}
    for detection in helper_payload.get("detections", []):
        detections_by_frame.setdefault(str(detection["frame_id"]), []).append(detection)

    dbscan_cfg = config.get("dbscan", {})
    run_records = []
    inference_csv_rows = []
    pointpillars_detections_json = output_dir / "pointpillars_inference_detections.json"
    pointpillars_bev_path = figures_dir / "pointpillars_bev_boxes.png"
    compare_path = figures_dir / "dbscan_vs_pointpillars_bev.png"
    projection_path = figures_dir / "pointpillars_camera_projection.png"
    latency_csv_path = output_dir / "latency_profile.csv"
    latency_json_path = output_dir / "latency_profile.json"
    online_latency_csv_path = output_dir / "online_latency_profile.csv"
    online_latency_json_path = output_dir / "online_latency_profile.json"
    inference_csv_path = output_dir / "pointpillars_inference_results.csv"
    bar_figure_path = figures_dir / "latency_breakdown.png"
    online_bar_figure_path = figures_dir / "online_latency_breakdown.png"

    first_measured_frame = None
    first_pointpillars_boxes = None
    first_dbscan_boxes = None
    first_points_xyz = None
    first_calibration = None
    first_image = None

    for helper_record in helper_payload.get("measured_records", []):
        frame_id = str(helper_record["frame_id"])
        assets = assets_by_frame[frame_id]

        start = time.perf_counter()
        calibration = parse_calibration_file(assets.calib_path) if assets.calib_path else None
        calibration_parse_ms = (time.perf_counter() - start) * 1000.0

        points = np.fromfile(str(assets.lidar_path), dtype=np.float32).reshape(-1, 4)
        points_xyz = points[:, :3]
        image = read_image_bgr(assets.image_path)
        pointpillars_boxes = [to_detection_box(item) for item in detections_by_frame.get(frame_id, [])]
        dbscan_result = build_dbscan_baseline(
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
        )
        dbscan_boxes = dbscan_result["boxes"]

        vis_start = time.perf_counter()
        _ = render_bev(points_xyz, boxes=pointpillars_boxes, title=f"PointPillars {frame_id}")
        if image is not None and calibration is not None:
            _ = draw_lidar_boxes_on_image(image, pointpillars_boxes, calibration, title=f"PointPillars Projection {frame_id}", source_name="pointpillars")
        visualization_ms = (time.perf_counter() - vis_start) * 1000.0

        total_ms = (
            float(helper_record["data_load_ms"])
            + calibration_parse_ms
            + float(helper_record["point_preprocess_ms"])
            + float(helper_record["voxelization_or_pillarization_ms"])
            + float(helper_record["model_forward_ms"])
            + float(helper_record["nms_ms"])
            + float(helper_record["postprocess_ms"])
            + visualization_ms
        )

        run_record = {
            "frame_id": frame_id,
            "data_load_ms": float(helper_record["data_load_ms"]),
            "calibration_parse_ms": calibration_parse_ms,
            "point_preprocess_ms": float(helper_record["point_preprocess_ms"]),
            "voxelization_or_pillarization_ms": float(helper_record["voxelization_or_pillarization_ms"]),
            "model_forward_ms": float(helper_record["model_forward_ms"]),
            "backbone_ms": float(helper_record["backbone_ms"]),
            "head_ms": float(helper_record["head_ms"]),
            "nms_ms": float(helper_record["nms_ms"]),
            "postprocess_ms": float(helper_record["postprocess_ms"]),
            "visualization_ms": visualization_ms,
            "total_ms": total_ms,
            "pillar_count": int(helper_record["pillar_count"]),
            "detected_box_count": int(helper_record["detected_box_count"]),
            "score_distribution": helper_record.get("score_distribution", []),
            "module_times": helper_record.get("module_times", {}),
        }
        run_records.append(run_record)

        for detection in detections_by_frame.get(frame_id, []):
            box = detection["box_3d_lidar"]
            inference_csv_rows.append(
                {
                    "frame_id": frame_id,
                    "object_id": detection["object_id"],
                    "class_name": detection["class_name"],
                    "score": detection["score"],
                    "center_x": box["x"],
                    "center_y": box["y"],
                    "center_z": box["z"],
                    "size_x": box["dx"],
                    "size_y": box["dy"],
                    "size_z": box["dz"],
                    "yaw": box["heading"],
                    "source": "pointpillars",
                }
            )

        if first_measured_frame is None:
            first_measured_frame = frame_id
            first_pointpillars_boxes = pointpillars_boxes
            first_dbscan_boxes = dbscan_boxes
            first_points_xyz = points_xyz
            first_calibration = calibration
            first_image = image

    aggregate_rows = aggregate_stage_records(run_records, STAGE_NAMES)
    write_csv(latency_csv_path, aggregate_rows, ["stage", "count", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"])
    write_csv(
        inference_csv_path,
        inference_csv_rows,
        ["frame_id", "object_id", "class_name", "score", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "yaw", "source"],
    )
    write_json(pointpillars_detections_json, {"detections": helper_payload.get("detections", []), "runtime_payload": helper_payload})
    write_json(
        latency_json_path,
        {
            "profile_mode": "full_inference",
            "warmup_runs": args.warmup,
            "measured_runs": len(run_records),
            "probe": probe.__dict__,
            "runtime_probe": runtime_probe,
            "stage_summary": aggregate_rows,
            "run_records": run_records,
            "model_build_ms": helper_payload.get("model_build_ms"),
            "checkpoint_load_ms": helper_payload.get("checkpoint_load_ms"),
        },
    )
    save_image(bar_figure_path, draw_bar_chart(aggregate_rows, title="Latency Breakdown (PointPillars CUDA)"))
    online_records = compute_online_debug_records(run_records)
    online_payload = summarize_online_latency(online_records)
    online_rows = online_payload["online_summary"] + online_payload["debug_summary"]
    write_csv(online_latency_csv_path, online_rows, ["stage", "count", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"])
    write_json(online_latency_json_path, online_payload)
    save_image(online_bar_figure_path, draw_bar_chart(online_payload["online_summary"], title="Online Perception Latency (Visualization Excluded)"))

    if first_measured_frame is not None:
        save_image(
            pointpillars_bev_path,
            render_bev(
                first_points_xyz,
                boxes=first_pointpillars_boxes,
                title=f"PointPillars BEV {first_measured_frame}",
                extra_lines=[
                    f"detections={len(first_pointpillars_boxes)}",
                    f"avg_score={statistics.fmean([box.score for box in first_pointpillars_boxes if box.score is not None]) if any(box.score is not None for box in first_pointpillars_boxes) else 0.0:.3f}",
                ],
            ),
        )
        save_image(
            compare_path,
            render_bev(
                first_points_xyz,
                boxes=list(first_dbscan_boxes) + list(first_pointpillars_boxes),
                title=f"DBSCAN vs PointPillars {first_measured_frame}",
                extra_lines=[f"dbscan={len(first_dbscan_boxes)}", f"pointpillars={len(first_pointpillars_boxes)}"],
            ),
        )
        if first_image is not None and first_calibration is not None:
            save_image(
                projection_path,
                draw_lidar_boxes_on_image(first_image, first_pointpillars_boxes, first_calibration, title=f"PointPillars Projection {first_measured_frame}", source_name="pointpillars"),
            )

    print(f"Saved latency profile: {latency_json_path}")
    print(f"Saved online latency profile: {online_latency_json_path}")
    print(f"Saved inference detections JSON: {pointpillars_detections_json}")
    print(f"Saved inference status CSV: {inference_csv_path}")


if __name__ == "__main__":
    main()

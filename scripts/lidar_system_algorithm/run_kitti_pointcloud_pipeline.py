from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.calibration import parse_calibration_file
from runtime.lidar_system_algorithm.dbscan_baseline import build_dbscan_baseline
from runtime.lidar_system_algorithm.kitti_io import locate_default_kitti_root, read_image_bgr, read_kitti_bin, resolve_frame_assets, save_image_bgr
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json
from runtime.lidar_system_algorithm.transforms import compute_distance_and_azimuth, filter_points_in_image, project_lidar_to_image
from runtime.lidar_system_algorithm.visualization import draw_box_centers_on_image, placeholder_image, render_bev, render_projection_image, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KITTI point cloud + calibration + DBSCAN pipeline.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--frame-id", default="000000", help="Frame id to load.")
    parser.add_argument("--config", default="projects/lidar_system_algorithm/configs/default.yaml", help="Project config path.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Report output directory.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload


def main() -> None:
    args = parse_args()
    kitti_root = Path(args.kitti_root).expanduser() if args.kitti_root else locate_default_kitti_root()
    config = load_config((PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = resolve_frame_assets(kitti_root, args.frame_id)
    points = read_kitti_bin(assets.lidar_path)
    points_xyz = points[:, :3]
    intensities = points[:, 3]
    image = read_image_bgr(assets.image_path)

    summary = {
        "frame_id": assets.frame_id,
        "dataset_style": assets.dataset_style,
        "lidar_path": str(assets.lidar_path),
        "image_path": str(assets.image_path) if assets.image_path else None,
        "calib_path": str(assets.calib_path) if assets.calib_path else None,
        "label_path": str(assets.label_path) if assets.label_path else None,
        "point_count": int(points.shape[0]),
        "projection_status": "skipped_missing_calibration",
        "dbscan_status": "not_run",
    }

    bev_path = figures_dir / "kitti_bev_example.png"
    projection_path = figures_dir / "kitti_projection_example.png"
    dbscan_bev_path = figures_dir / "dbscan_bev_boxes.png"
    dbscan_proj_path = figures_dir / "dbscan_camera_projection.png"
    summary_path = output_dir / "kitti_pipeline_summary.json"
    dbscan_csv_path = output_dir / "dbscan_baseline_results.csv"

    if assets.calib_path is None or not assets.calib_path.exists():
        save_image(bev_path, render_bev(points_xyz, title=f"KITTI BEV {assets.frame_id}"))
        save_image(projection_path, placeholder_image(["Calibration file missing", f"frame_id={assets.frame_id}"]))
        write_json(summary_path, summary)
        write_csv(dbscan_csv_path, [], ["frame_id", "cluster_id", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "distance_m", "azimuth_deg", "point_count", "score", "source"])
        raise SystemExit("Calibration file missing. BEV was still generated; projection was skipped.")

    calibration = parse_calibration_file(assets.calib_path)
    rectified, uv, valid_projection = project_lidar_to_image(points_xyz, calibration)
    if image is not None:
        in_image_mask = filter_points_in_image(uv, valid_projection, image.shape)
        projection_image = render_projection_image(
            image_bgr=image,
            uv=uv[in_image_mask],
            depths=rectified[in_image_mask, 2],
            title=f"KITTI Projection {assets.frame_id}",
        )
        save_image(projection_path, projection_image)
        projected_count = int(in_image_mask.sum())
        summary["image_shape"] = list(image.shape)
    else:
        save_image(projection_path, placeholder_image(["Image missing", f"frame_id={assets.frame_id}"]))
        projected_count = 0

    distances, azimuths = compute_distance_and_azimuth(points_xyz)
    fov_mask = (points_xyz[:, 0] > 0.0) & (abs(azimuths) <= 45.0)
    summary.update(
        {
            "projection_status": "ok" if image is not None else "skipped_missing_image",
            "projected_point_count": projected_count,
            "rectified_camera_point_count": int(valid_projection.sum()),
            "fov_point_count": int(fov_mask.sum()),
            "distance_range_m": [round(float(distances.min()), 6), round(float(distances.max()), 6)],
            "azimuth_range_deg": [round(float(azimuths.min()), 6), round(float(azimuths.max()), 6)],
            "intensity_range": [round(float(intensities.min()), 6), round(float(intensities.max()), 6)],
        }
    )

    save_image(bev_path, render_bev(points_xyz, title=f"KITTI BEV {assets.frame_id}", extra_lines=[f"points={points.shape[0]}", f"fov_points={int(fov_mask.sum())}"]))

    dbscan_cfg = config.get("dbscan", {})
    roi = dbscan_cfg.get("roi", {})
    dbscan_result = build_dbscan_baseline(
        points_xyz=points_xyz,
        roi=(
            tuple(roi.get("x_range", [0.0, 60.0])),
            tuple(roi.get("y_range", [-25.0, 25.0])),
            tuple(roi.get("z_range", [-2.5, 1.5])),
        ),
        ground_z_threshold=dbscan_cfg.get("ground_z_threshold", -1.35),
        eps=float(dbscan_cfg.get("eps", 0.8)),
        min_points=int(dbscan_cfg.get("min_points", 12)),
        min_cluster_points=int(dbscan_cfg.get("min_cluster_points", 20)),
    )
    boxes = dbscan_result["boxes"]
    dbscan_rows = [box.to_row(assets.frame_id) for box in boxes]
    write_csv(
        dbscan_csv_path,
        dbscan_rows,
        ["frame_id", "cluster_id", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "distance_m", "azimuth_deg", "point_count", "score", "source"],
    )
    save_image(
        dbscan_bev_path,
        render_bev(
            dbscan_result["filtered_points"],
            boxes=boxes,
            title=f"DBSCAN BEV {assets.frame_id}",
            extra_lines=[
                f"roi_points={dbscan_result['metadata']['roi_point_count']}",
                f"filtered_points={dbscan_result['metadata']['filtered_point_count']}",
                f"clusters={dbscan_result['metadata']['cluster_count']}",
            ],
        ),
    )
    if image is not None:
        save_image(dbscan_proj_path, draw_box_centers_on_image(image, boxes, calibration, title="DBSCAN center projection", source_name="dbscan"))
    else:
        save_image(dbscan_proj_path, placeholder_image(["Image missing", "DBSCAN camera projection skipped"]))
    summary.update(
        {
            "dbscan_status": "ok",
            "dbscan_metadata": dbscan_result["metadata"],
            "dbscan_cluster_count": len(boxes),
            "dbscan_csv": str(dbscan_csv_path),
            "projection_figure": str(projection_path),
            "bev_figure": str(bev_path),
            "dbscan_bev_figure": str(dbscan_bev_path),
        }
    )
    write_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")
    print(f"Saved projection figure: {projection_path}")
    print(f"Saved BEV figure: {bev_path}")
    print(f"Saved DBSCAN CSV: {dbscan_csv_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime.lidar_system_algorithm.kitti_io import list_frame_ids, locate_default_kitti_root, read_kitti_bin, resolve_frame_assets
from runtime.lidar_system_algorithm.report_schema import write_csv, write_json
from runtime.lidar_system_algorithm.visualization import draw_tradeoff_chart, save_image
from runtime.lidar_system_algorithm.voxelization_ablation import pillarize_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight pillarization ablation.")
    parser.add_argument("--kitti-root", default="", help="KITTI root directory.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames.")
    parser.add_argument("--pillar-sizes", default="0.12,0.16,0.20,0.24", help="Comma separated pillar sizes.")
    parser.add_argument("--max-points-per-voxel", type=int, default=32, help="Maximum points per pillar.")
    parser.add_argument("--max-voxels", type=int, default=12000, help="Maximum number of pillars.")
    parser.add_argument("--output-dir", default="reports/lidar_system_algorithm", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kitti_root = Path(args.kitti_root).expanduser() if args.kitti_root else locate_default_kitti_root()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    figures_dir = PROJECT_ROOT / "projects" / "lidar_system_algorithm" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = list_frame_ids(kitti_root, limit=args.frames)
    if not frame_ids:
        raise SystemExit(f"No KITTI frames found under: {kitti_root}")

    pillar_sizes = [float(token.strip()) for token in args.pillar_sizes.split(",") if token.strip()]
    point_cloud_range = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)

    rows: list[dict] = []
    detailed: list[dict] = []
    for pillar_size in pillar_sizes:
        preprocess_times: list[float] = []
        pillar_counts: list[int] = []
        kept_counts: list[int] = []
        memory_bytes: list[int] = []
        for frame_id in frame_ids:
            assets = resolve_frame_assets(kitti_root, frame_id)
            points = read_kitti_bin(assets.lidar_path)[:, :3]
            start = time.perf_counter()
            stats = pillarize_points(
                points_xyz=points,
                pillar_size=pillar_size,
                point_cloud_range=point_cloud_range,
                max_points_per_voxel=args.max_points_per_voxel,
                max_voxels=args.max_voxels,
            )
            preprocess_times.append((time.perf_counter() - start) * 1000.0)
            pillar_counts.append(int(stats["pillar_count"]))
            kept_counts.append(int(stats["kept_point_count"]))
            memory_bytes.append(int(stats["memory_bytes_estimate"]))
            detailed.append(
                {
                    "frame_id": frame_id,
                    "pillar_size": pillar_size,
                    "pillar_count": stats["pillar_count"],
                    "kept_point_count": stats["kept_point_count"],
                    "memory_bytes_estimate": stats["memory_bytes_estimate"],
                    "preprocess_ms": preprocess_times[-1],
                }
            )
        rows.append(
            {
                "pillar_size": pillar_size,
                "frame_count": len(frame_ids),
                "pillar_count_mean": statistics.fmean(pillar_counts),
                "pillar_count_min": min(pillar_counts),
                "pillar_count_max": max(pillar_counts),
                "kept_point_count_mean": statistics.fmean(kept_counts),
                "preprocess_mean_ms": statistics.fmean(preprocess_times),
                "preprocess_p95_ms": sorted(preprocess_times)[int(0.95 * (len(preprocess_times) - 1))] if len(preprocess_times) > 1 else preprocess_times[0],
                "memory_bytes_mean": statistics.fmean(memory_bytes),
                "model_forward_ms": "",
                "nms_ms": "",
                "total_latency_mode": "preprocess_only",
                "detected_box_count": "",
                "score_distribution": "unavailable_without_model",
            }
        )

    csv_path = output_dir / "voxelization_ablation.csv"
    json_path = output_dir / "voxelization_ablation.json"
    figure_path = figures_dir / "voxelization_latency_tradeoff.png"
    write_csv(
        csv_path,
        rows,
        [
            "pillar_size",
            "frame_count",
            "pillar_count_mean",
            "pillar_count_min",
            "pillar_count_max",
            "kept_point_count_mean",
            "preprocess_mean_ms",
            "preprocess_p95_ms",
            "memory_bytes_mean",
            "model_forward_ms",
            "nms_ms",
            "total_latency_mode",
            "detected_box_count",
            "score_distribution",
        ],
    )
    write_json(
        json_path,
        {
            "mode": "preprocess_only",
            "reason": "Checkpoint-dependent latency and score statistics were intentionally skipped.",
            "summary_rows": rows,
            "per_frame_rows": detailed,
        },
    )
    save_image(figure_path, draw_tradeoff_chart(rows, "Pillar Size vs Preprocess Latency / Pillar Count"))
    print(f"Saved voxelization ablation: {json_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np


def pillarize_points(
    points_xyz: np.ndarray,
    pillar_size: float,
    point_cloud_range: tuple[float, float, float, float, float, float] = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
    max_points_per_voxel: int = 32,
    max_voxels: int = 12000,
) -> dict:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    mask = (
        (points_xyz[:, 0] >= x_min)
        & (points_xyz[:, 0] < x_max)
        & (points_xyz[:, 1] >= y_min)
        & (points_xyz[:, 1] < y_max)
        & (points_xyz[:, 2] >= z_min)
        & (points_xyz[:, 2] < z_max)
    )
    filtered = points_xyz[mask]
    if filtered.size == 0:
        return {
            "input_point_count": int(points_xyz.shape[0]),
            "kept_point_count": 0,
            "pillar_count": 0,
            "mean_points_per_pillar": 0.0,
            "max_points_per_pillar_observed": 0,
            "memory_bytes_estimate": 0,
            "occupancies": [],
        }

    x_idx = np.floor((filtered[:, 0] - x_min) / pillar_size).astype(np.int32)
    y_idx = np.floor((filtered[:, 1] - y_min) / pillar_size).astype(np.int32)
    occupancies: dict[tuple[int, int], int] = {}
    for ix, iy in zip(x_idx.tolist(), y_idx.tolist()):
        key = (ix, iy)
        if key not in occupancies and len(occupancies) >= max_voxels:
            continue
        occupancies[key] = min(max_points_per_voxel, occupancies.get(key, 0) + 1)

    counts = list(occupancies.values())
    memory_bytes = len(counts) * max_points_per_voxel * 4 * 4
    return {
        "input_point_count": int(points_xyz.shape[0]),
        "kept_point_count": int(filtered.shape[0]),
        "pillar_count": len(counts),
        "mean_points_per_pillar": float(np.mean(counts)) if counts else 0.0,
        "max_points_per_pillar_observed": int(max(counts)) if counts else 0,
        "memory_bytes_estimate": int(memory_bytes),
        "occupancies": counts,
    }

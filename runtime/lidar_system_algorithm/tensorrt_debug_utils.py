from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def numeric_summary(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    seq = [float(value) for value in values if value is not None]
    if not seq:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None, "std": None}
    seq.sort()
    array = np.asarray(seq, dtype=np.float64)

    def percentile(pct: float) -> float:
        return float(np.percentile(array, pct))

    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "p50": percentile(50.0),
        "p95": percentile(95.0),
        "std": float(array.std()),
    }


def summarize_tensor(array: np.ndarray) -> dict[str, float | int | list[int]]:
    array = np.asarray(array)
    finite_mask = np.isfinite(array)
    finite_values = array[finite_mask]
    zero_ratio = float(np.mean(array == 0)) if array.size else 0.0
    if finite_values.size == 0:
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": int(np.isnan(array).sum()),
            "inf_count": int(np.isinf(array).sum()),
            "zero_ratio": zero_ratio,
        }
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std()),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "zero_ratio": zero_ratio,
    }


def summarize_diff(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | int | list[int] | None]:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        return {
            "shape_lhs": list(lhs.shape),
            "shape_rhs": list(rhs.shape),
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "p99_abs_diff": None,
            "nan_count": None,
            "inf_count": None,
            "zero_ratio": None,
            "status": "shape_mismatch",
        }
    diff = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    finite = diff[np.isfinite(diff)]
    return {
        "shape": list(diff.shape),
        "max_abs_diff": float(np.max(finite)) if finite.size else math.inf,
        "mean_abs_diff": float(np.mean(finite)) if finite.size else math.inf,
        "p99_abs_diff": float(np.percentile(finite, 99.0)) if finite.size else math.inf,
        "nan_count": int(np.isnan(diff).sum()),
        "inf_count": int(np.isinf(diff).sum()),
        "zero_ratio": float(np.mean(diff == 0)) if diff.size else 0.0,
        "status": "completed",
    }


def _ensure_bucket_arrays(prepared: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return prepared["voxels"].copy(), prepared["voxel_num_points"].copy(), prepared["voxel_coords"].copy()


def _grid_shape_from_coords(coords: np.ndarray) -> tuple[int, int]:
    if coords.size == 0:
        return 496, 432
    y_index = 2 if coords.shape[1] >= 4 else 1
    x_index = 3 if coords.shape[1] >= 4 else 2
    y_max = int(coords[:, y_index].max()) + 1
    x_max = int(coords[:, x_index].max()) + 1
    return max(496, y_max), max(432, x_max)


def _build_unique_dummy_coords(existing_coords: np.ndarray, count: int) -> tuple[np.ndarray, int]:
    grid_y, grid_x = _grid_shape_from_coords(existing_coords)
    y_index = 2 if existing_coords.shape[1] >= 4 else 1
    x_index = 3 if existing_coords.shape[1] >= 4 else 2
    used = {(int(coord[y_index]), int(coord[x_index])) for coord in existing_coords}
    dummy_coords = []
    collision_count = 0
    coord_width = int(existing_coords.shape[1]) if existing_coords.ndim == 2 else 4
    for y_index in range(grid_y):
        for x_index in range(grid_x):
            key = (y_index, x_index)
            if key in used:
                continue
            if coord_width >= 4:
                dummy_coords.append([0, 0, y_index, x_index])
            else:
                dummy_coords.append([0, y_index, x_index])
            if len(dummy_coords) >= count:
                return np.asarray(dummy_coords, dtype=existing_coords.dtype), collision_count
    collision_count = count - len(dummy_coords)
    if not dummy_coords:
        return np.zeros((0, 4), dtype=existing_coords.dtype), collision_count
    pad = np.repeat(np.asarray(dummy_coords[:1], dtype=existing_coords.dtype), collision_count, axis=0)
    dummy = np.concatenate([np.asarray(dummy_coords, dtype=existing_coords.dtype), pad], axis=0)
    return dummy, collision_count


def pad_prepared_inputs(prepared: dict, bucket_size: int, strategy: str) -> tuple[dict, dict]:
    voxels, voxel_num_points, voxel_coords = _ensure_bucket_arrays(prepared)
    current_count = int(voxels.shape[0])
    if current_count >= bucket_size:
        capped = dict(prepared)
        capped["voxels"] = voxels[:bucket_size]
        capped["voxel_num_points"] = voxel_num_points[:bucket_size]
        capped["voxel_coords"] = voxel_coords[:bucket_size]
        return capped, {
            "strategy": strategy,
            "real_pillar_count": current_count,
            "padded_pillar_count": 0,
            "duplicate_coord_count": 0,
            "dummy_coord_collision_count": 0,
            "status": "no_padding",
        }

    pad_count = bucket_size - current_count
    capped = dict(prepared)
    if current_count <= 0:
        capped["voxels"] = np.zeros((bucket_size,) + voxels.shape[1:], dtype=voxels.dtype)
        capped["voxel_num_points"] = np.ones((bucket_size,), dtype=voxel_num_points.dtype)
        capped["voxel_coords"] = np.zeros((bucket_size,) + voxel_coords.shape[1:], dtype=voxel_coords.dtype)
        return capped, {
            "strategy": strategy,
            "real_pillar_count": current_count,
            "padded_pillar_count": pad_count,
            "duplicate_coord_count": pad_count,
            "dummy_coord_collision_count": 0,
            "status": "empty_input",
        }

    duplicate_coord_count = 0
    dummy_coord_collision_count = 0
    if strategy == "repeat_first_valid":
        pad_voxels = np.repeat(voxels[:1], pad_count, axis=0)
        pad_num_points = np.repeat(voxel_num_points[:1], pad_count, axis=0)
        pad_coords = np.repeat(voxel_coords[:1], pad_count, axis=0)
        duplicate_coord_count = pad_count
    elif strategy == "duplicate_zero_coord_padding":
        pad_voxels = np.zeros((pad_count,) + voxels.shape[1:], dtype=voxels.dtype)
        pad_num_points = np.ones((pad_count,), dtype=voxel_num_points.dtype)
        pad_coords = np.zeros((pad_count,) + voxel_coords.shape[1:], dtype=voxel_coords.dtype)
        duplicate_coord_count = pad_count
    elif strategy == "unique_dummy_coord_padding":
        pad_voxels = np.zeros((pad_count,) + voxels.shape[1:], dtype=voxels.dtype)
        pad_num_points = np.ones((pad_count,), dtype=voxel_num_points.dtype)
        pad_coords, dummy_coord_collision_count = _build_unique_dummy_coords(voxel_coords, pad_count)
    else:
        raise ValueError(f"Unsupported padding strategy: {strategy}")

    capped["voxels"] = np.concatenate([voxels, pad_voxels], axis=0)
    capped["voxel_num_points"] = np.concatenate([voxel_num_points, pad_num_points], axis=0)
    capped["voxel_coords"] = np.concatenate([voxel_coords, pad_coords], axis=0)
    return capped, {
        "strategy": strategy,
        "real_pillar_count": current_count,
        "padded_pillar_count": pad_count,
        "duplicate_coord_count": duplicate_coord_count,
        "dummy_coord_collision_count": dummy_coord_collision_count,
        "status": "completed",
    }


def prediction_dir_stats(pred_dir: Path) -> dict:
    files = sorted(pred_dir.glob("*.txt"))
    empty = [path.name for path in files if not path.read_text(encoding="utf-8", errors="ignore").strip()]
    total_lines = 0
    scores = []
    class_counts: dict[str, int] = {}
    invalid_geometry_count = 0
    invalid_line_count = 0
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 16:
                invalid_line_count += 1
                continue
            total_lines += 1
            class_counts[parts[0]] = class_counts.get(parts[0], 0) + 1
            try:
                scores.append(float(parts[15]))
            except Exception:
                invalid_line_count += 1
                continue
            try:
                height = float(parts[8])
                width = float(parts[9])
                length = float(parts[10])
                location = [float(parts[11]), float(parts[12]), float(parts[13])]
                rotation_y = float(parts[14])
                values = [height, width, length, rotation_y, *location]
                if height <= 0 or width <= 0 or length <= 0 or not all(np.isfinite(values)):
                    invalid_geometry_count += 1
            except Exception:
                invalid_geometry_count += 1
    return {
        "prediction_file_count": len(files),
        "empty_prediction_file_count": len(empty),
        "empty_prediction_files_preview": empty[:10],
        "total_box_count": total_lines,
        "per_class_box_count": class_counts,
        "score_summary": numeric_summary(scores),
        "invalid_geometry_count": invalid_geometry_count,
        "invalid_line_count": invalid_line_count,
    }

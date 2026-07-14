from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionBox:
    cluster_id: int
    center_xyz: np.ndarray
    size_xyz: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    point_count: int
    distance_m: float
    azimuth_deg: float
    score: float | None = None
    class_name: str = "cluster"
    yaw: float = 0.0
    source: str = "dbscan"

    def to_row(self, frame_id: str) -> dict:
        return {
            "frame_id": frame_id,
            "cluster_id": self.cluster_id,
            "center_x": round(float(self.center_xyz[0]), 6),
            "center_y": round(float(self.center_xyz[1]), 6),
            "center_z": round(float(self.center_xyz[2]), 6),
            "size_x": round(float(self.size_xyz[0]), 6),
            "size_y": round(float(self.size_xyz[1]), 6),
            "size_z": round(float(self.size_xyz[2]), 6),
            "distance_m": round(float(self.distance_m), 6),
            "azimuth_deg": round(float(self.azimuth_deg), 6),
            "point_count": self.point_count,
            "score": "" if self.score is None else round(float(self.score), 6),
            "source": self.source,
        }


def crop_roi(
    points: np.ndarray,
    x_range: tuple[float, float] = (0.0, 60.0),
    y_range: tuple[float, float] = (-25.0, 25.0),
    z_range: tuple[float, float] = (-2.5, 1.5),
) -> np.ndarray:
    mask = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0])
        & (points[:, 1] <= y_range[1])
        & (points[:, 2] >= z_range[0])
        & (points[:, 2] <= z_range[1])
    )
    return points[mask]


def remove_ground(points: np.ndarray, ground_z_threshold: float = -1.35) -> np.ndarray:
    return points[points[:, 2] > ground_z_threshold]


def _grid_key(x: float, y: float, cell_size: float) -> tuple[int, int]:
    return int(math.floor(x / cell_size)), int(math.floor(y / cell_size))


def _build_grid(points_xy: np.ndarray, eps: float) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for index, (x, y) in enumerate(points_xy):
        key = _grid_key(float(x), float(y), eps)
        grid.setdefault(key, []).append(index)
    return grid


def _region_query(points_xy: np.ndarray, grid: dict[tuple[int, int], list[int]], eps: float, index: int) -> list[int]:
    x, y = points_xy[index]
    key_x, key_y = _grid_key(float(x), float(y), eps)
    eps_sq = eps * eps
    neighbors: list[int] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for candidate in grid.get((key_x + dx, key_y + dy), []):
                delta = points_xy[candidate] - points_xy[index]
                if float(delta[0] * delta[0] + delta[1] * delta[1]) <= eps_sq:
                    neighbors.append(candidate)
    return neighbors


def dbscan_cluster(points_xyz: np.ndarray, eps: float = 0.8, min_points: int = 12) -> np.ndarray:
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=np.int32)
    points_xy = points_xyz[:, :2].astype(np.float64)
    grid = _build_grid(points_xy, eps)
    labels = np.full(points_xyz.shape[0], -2, dtype=np.int32)
    cluster_id = 0
    for index in range(points_xyz.shape[0]):
        if labels[index] != -2:
            continue
        neighbors = _region_query(points_xy, grid, eps, index)
        if len(neighbors) < min_points:
            labels[index] = -1
            continue
        labels[index] = cluster_id
        queue: deque[int] = deque(neighbors)
        while queue:
            current = queue.popleft()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -2:
                continue
            labels[current] = cluster_id
            current_neighbors = _region_query(points_xy, grid, eps, current)
            if len(current_neighbors) >= min_points:
                queue.extend(current_neighbors)
        cluster_id += 1
    return labels


def labels_to_boxes(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    min_cluster_points: int = 20,
) -> list[DetectionBox]:
    boxes: list[DetectionBox] = []
    for cluster_id in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
        cluster_points = points_xyz[labels == cluster_id]
        if cluster_points.shape[0] < min_cluster_points:
            continue
        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)
        center_xyz = (min_xyz + max_xyz) / 2.0
        size_xyz = np.maximum(max_xyz - min_xyz, 1e-3)
        distance_m = float(np.linalg.norm(center_xyz[:3]))
        azimuth_deg = float(np.degrees(np.arctan2(center_xyz[1], center_xyz[0])))
        score = min(1.0, cluster_points.shape[0] / 120.0)
        boxes.append(
            DetectionBox(
                cluster_id=cluster_id,
                center_xyz=center_xyz,
                size_xyz=size_xyz,
                min_xyz=min_xyz,
                max_xyz=max_xyz,
                point_count=int(cluster_points.shape[0]),
                distance_m=distance_m,
                azimuth_deg=azimuth_deg,
                score=score,
            )
        )
    return boxes


def build_dbscan_baseline(
    points_xyz: np.ndarray,
    roi: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((0.0, 60.0), (-25.0, 25.0), (-2.5, 1.5)),
    ground_z_threshold: float | None = -1.35,
    eps: float = 0.8,
    min_points: int = 12,
    min_cluster_points: int = 20,
    sample_step: int = 1,
) -> dict:
    cropped = crop_roi(points_xyz, x_range=roi[0], y_range=roi[1], z_range=roi[2])
    filtered = remove_ground(cropped, ground_z_threshold) if ground_z_threshold is not None else cropped
    if sample_step > 1:
        filtered = filtered[::sample_step]
    labels = dbscan_cluster(filtered, eps=eps, min_points=min_points)
    boxes = labels_to_boxes(filtered, labels, min_cluster_points=min_cluster_points)
    return {
        "roi_points": cropped,
        "filtered_points": filtered,
        "labels": labels,
        "boxes": boxes,
        "metadata": {
            "roi_point_count": int(cropped.shape[0]),
            "filtered_point_count": int(filtered.shape[0]),
            "cluster_count": len(boxes),
            "eps": eps,
            "min_points": min_points,
            "min_cluster_points": min_cluster_points,
            "ground_z_threshold": ground_z_threshold,
            "sample_step": sample_step,
            "roi": {
                "x_range": list(roi[0]),
                "y_range": list(roi[1]),
                "z_range": list(roi[2]),
            },
        },
    }

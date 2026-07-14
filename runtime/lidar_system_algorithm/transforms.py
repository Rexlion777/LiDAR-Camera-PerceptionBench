from __future__ import annotations

import math

import numpy as np

from .calibration import Calibration


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("Points must have shape (N, C)")
    return np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)


def lidar_to_rectified_camera(points_xyz: np.ndarray, calibration: Calibration) -> np.ndarray:
    points_h = to_homogeneous(points_xyz)
    camera = (calibration.tr_velo_to_cam @ points_h.T).T
    rectified = (calibration.r0_rect @ camera.T).T
    return rectified


def project_rectified_to_image(points_rect: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    points_h = to_homogeneous(points_rect)
    uvw = (calibration.p2 @ points_h.T).T
    valid = uvw[:, 2] > 1e-6
    uv = np.zeros((points_rect.shape[0], 2), dtype=np.float64)
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, valid


def project_lidar_to_image(points_xyz: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rectified = lidar_to_rectified_camera(points_xyz, calibration)
    uv, valid = project_rectified_to_image(rectified, calibration)
    return rectified, uv, valid


def filter_points_in_image(
    uv: np.ndarray,
    valid_mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> np.ndarray:
    image_height, image_width = image_shape[:2]
    return (
        valid_mask
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(image_width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(image_height))
    )


def compute_distance_and_azimuth(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(points_xyz[:, :3], axis=1)
    azimuth_deg = np.degrees(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
    return distances, azimuth_deg


def axis_aligned_box_corners(center_xyz: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    cx, cy, cz = center_xyz.astype(np.float64)
    dx, dy, dz = size_xyz.astype(np.float64)
    offsets = np.array(
        [
            [dx / 2.0, dy / 2.0, dz / 2.0],
            [dx / 2.0, -dy / 2.0, dz / 2.0],
            [-dx / 2.0, -dy / 2.0, dz / 2.0],
            [-dx / 2.0, dy / 2.0, dz / 2.0],
            [dx / 2.0, dy / 2.0, -dz / 2.0],
            [dx / 2.0, -dy / 2.0, -dz / 2.0],
            [-dx / 2.0, -dy / 2.0, -dz / 2.0],
            [-dx / 2.0, dy / 2.0, -dz / 2.0],
        ],
        dtype=np.float64,
    )
    return offsets + np.array([cx, cy, cz], dtype=np.float64)


def oriented_lidar_box_corners(center_xyz: np.ndarray, size_xyz: np.ndarray, yaw: float) -> np.ndarray:
    cx, cy, cz = center_xyz.astype(np.float64)
    dx, dy, dz = size_xyz.astype(np.float64)
    corners = np.array(
        [
            [dx / 2.0, dy / 2.0, dz / 2.0],
            [dx / 2.0, -dy / 2.0, dz / 2.0],
            [-dx / 2.0, -dy / 2.0, dz / 2.0],
            [-dx / 2.0, dy / 2.0, dz / 2.0],
            [dx / 2.0, dy / 2.0, -dz / 2.0],
            [dx / 2.0, -dy / 2.0, -dz / 2.0],
            [-dx / 2.0, -dy / 2.0, -dz / 2.0],
            [-dx / 2.0, dy / 2.0, -dz / 2.0],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated = corners @ rotation.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    rotated[:, 2] += cz
    return rotated


def bev_corners(center_x: float, center_y: float, dx: float, dy: float, heading: float = 0.0) -> np.ndarray:
    local = np.array(
        [
            [dx / 2.0, dy / 2.0],
            [dx / 2.0, -dy / 2.0],
            [-dx / 2.0, -dy / 2.0],
            [-dx / 2.0, dy / 2.0],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(heading)
    sine = math.sin(heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    corners = local @ rotation.T
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    return corners

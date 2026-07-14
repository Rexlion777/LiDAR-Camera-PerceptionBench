from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dbscan_baseline import DetectionBox
from .transforms import bev_corners, oriented_lidar_box_corners, project_rectified_to_image, lidar_to_rectified_camera


EDGE_INDICES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _color_bgr(name: str) -> tuple[int, int, int]:
    palette = {
        "dbscan": (0, 180, 255),
        "pointpillars": (0, 220, 80),
        "track": (255, 140, 0),
        "cluster": (0, 180, 255),
    }
    return palette.get(name, (40, 40, 220))


def placeholder_image(lines: list[str], width: int = 1200, height: int = 700) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(image, "Unavailable", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 180), 3, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (40, 140 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA)
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image for: {path}")
    encoded.tofile(str(path))


def render_projection_image(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depths: np.ndarray,
    title: str,
    point_radius: int = 2,
) -> np.ndarray:
    output = image_bgr.copy()
    if uv.shape[0] > 0:
        near = float(depths.min())
        far = float(depths.max())
        ordering = np.argsort(depths)[::-1]
        for index in ordering:
            ratio = 0.0 if far - near < 1e-6 else float((depths[index] - near) / (far - near))
            color = (
                int(255 * ratio),
                int(120 + 80 * (1.0 - ratio)),
                int(255 * (1.0 - ratio)),
            )
            cv2.circle(
                output,
                (int(round(float(uv[index, 0]))), int(round(float(uv[index, 1])))),
                point_radius,
                color,
                -1,
                cv2.LINE_AA,
            )
    cv2.putText(output, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 220), 2, cv2.LINE_AA)
    return output


def draw_box_centers_on_image(
    image_bgr: np.ndarray,
    boxes: list[DetectionBox],
    calibration,
    title: str,
    source_name: str,
) -> np.ndarray:
    output = image_bgr.copy()
    points_xyz = np.array([box.center_xyz for box in boxes], dtype=np.float64) if boxes else np.zeros((0, 3), dtype=np.float64)
    if points_xyz.shape[0] > 0:
        rectified = lidar_to_rectified_camera(points_xyz, calibration)
        uv, valid = project_rectified_to_image(rectified, calibration)
        for index, box in enumerate(boxes):
            if not valid[index]:
                continue
            u = int(round(float(uv[index, 0])))
            v = int(round(float(uv[index, 1])))
            if u < 0 or u >= output.shape[1] or v < 0 or v >= output.shape[0]:
                continue
            color = _color_bgr(source_name)
            cv2.circle(output, (u, v), 6, color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                f"{source_name}:{box.cluster_id}",
                (u + 8, max(20, v - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(output, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 220), 2, cv2.LINE_AA)
    return output


def draw_lidar_boxes_on_image(
    image_bgr: np.ndarray,
    boxes: list[DetectionBox],
    calibration,
    title: str,
    source_name: str,
) -> np.ndarray:
    output = image_bgr.copy()
    color = _color_bgr(source_name)
    for box in boxes:
        corners = oriented_lidar_box_corners(box.center_xyz, box.size_xyz, box.yaw)
        rectified = lidar_to_rectified_camera(corners, calibration)
        uv, valid = project_rectified_to_image(rectified, calibration)
        if int(valid.sum()) < 8:
            continue
        points = [(int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))) for i in range(8)]
        if all((u < 0 or u >= output.shape[1] or v < 0 or v >= output.shape[0]) for u, v in points):
            continue
        for start, end in EDGE_INDICES:
            cv2.line(output, points[start], points[end], color, 2, cv2.LINE_AA)
        anchor = points[0]
        cv2.putText(
            output,
            f"{box.class_name}:{'' if box.score is None else f'{box.score:.2f}'}",
            (anchor[0] + 6, max(18, anchor[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(output, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 220), 2, cv2.LINE_AA)
    return output


def render_bev(
    points_xyz: np.ndarray,
    boxes: list[DetectionBox] | None = None,
    title: str = "BEV",
    x_range: tuple[float, float] = (0.0, 60.0),
    y_range: tuple[float, float] = (-25.0, 25.0),
    width: int = 1000,
    height: int = 1000,
    extra_lines: list[str] | None = None,
    track_labels: dict[int, str] | None = None,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        px = int((y - y_range[0]) / (y_range[1] - y_range[0]) * (width - 1))
        py = int((1.0 - (x - x_range[0]) / (x_range[1] - x_range[0])) * (height - 1))
        return px, py

    if points_xyz.size > 0:
        mask = (
            (points_xyz[:, 0] >= x_range[0])
            & (points_xyz[:, 0] <= x_range[1])
            & (points_xyz[:, 1] >= y_range[0])
            & (points_xyz[:, 1] <= y_range[1])
        )
        for point in points_xyz[mask]:
            px, py = to_pixel(float(point[0]), float(point[1]))
            cv2.circle(canvas, (px, py), 1, (90, 90, 90), -1, cv2.LINE_AA)

    if boxes:
        for box in boxes:
            corners = bev_corners(
                center_x=float(box.center_xyz[0]),
                center_y=float(box.center_xyz[1]),
                dx=float(box.size_xyz[0]),
                dy=float(box.size_xyz[1]),
                heading=float(box.yaw),
            )
            polygon = np.array([to_pixel(float(x), float(y)) for x, y in corners], dtype=np.int32)
            color = _color_bgr(box.source)
            cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            center = to_pixel(float(box.center_xyz[0]), float(box.center_xyz[1]))
            cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
            label = f"{box.source}:{box.cluster_id}"
            if track_labels and box.cluster_id in track_labels:
                label = track_labels[box.cluster_id]
            cv2.putText(canvas, label, (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (120, 120, 120), 2)
    cv2.putText(canvas, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 220), 2, cv2.LINE_AA)
    if extra_lines:
        for index, line in enumerate(extra_lines[:8]):
            cv2.putText(canvas, line, (20, 70 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
    return canvas


def draw_bar_chart(rows: list[dict], title: str, width: int = 1200, height: int = 700) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    margin_left = 220
    margin_right = 60
    margin_top = 100
    margin_bottom = 90
    available = [row for row in rows if row.get("mean_ms") is not None]
    max_value = max((float(row["mean_ms"]) for row in available), default=1.0)
    bar_area_width = width - margin_left - margin_right
    bar_step = max(1, int((height - margin_top - margin_bottom) / max(len(rows), 1)))
    cv2.putText(canvas, title, (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 220), 2, cv2.LINE_AA)
    for index, row in enumerate(rows):
        y = margin_top + index * bar_step
        stage = str(row["stage"])
        mean_ms = row.get("mean_ms")
        cv2.putText(canvas, stage, (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
        if mean_ms is None:
            cv2.putText(canvas, "unavailable", (margin_left, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 180), 2, cv2.LINE_AA)
            continue
        bar_length = int(float(mean_ms) / max_value * bar_area_width)
        cv2.rectangle(canvas, (margin_left, y), (margin_left + bar_length, y + 24), (0, 170, 255), -1)
        cv2.putText(canvas, f"{float(mean_ms):.2f} ms", (margin_left + bar_length + 12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
    return canvas


def draw_tradeoff_chart(rows: list[dict], title: str, width: int = 1200, height: int = 700) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right, top, bottom = 100, 80, 90, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.putText(canvas, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 220), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (120, 120, 120), 2)
    if not rows:
        cv2.putText(canvas, "No data", (left + 40, top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 180), 2, cv2.LINE_AA)
        return canvas
    sizes = [float(row["pillar_size"]) for row in rows]
    latencies = [float(row["preprocess_mean_ms"]) for row in rows]
    counts = [float(row["pillar_count_mean"]) for row in rows]
    min_size, max_size = min(sizes), max(sizes)
    min_latency, max_latency = min(latencies), max(latencies)
    min_count, max_count = min(counts), max(counts)

    def norm(value: float, low: float, high: float) -> float:
        if abs(high - low) < 1e-6:
            return 0.5
        return (value - low) / (high - low)

    latency_points = []
    count_points = []
    for size, latency, count in zip(sizes, latencies, counts):
        x = left + int(norm(size, min_size, max_size) * plot_w)
        y_latency = top + plot_h - int(norm(latency, min_latency, max_latency) * plot_h)
        y_count = top + plot_h - int(norm(count, min_count, max_count) * plot_h)
        latency_points.append((x, y_latency))
        count_points.append((x, y_count))
        cv2.putText(canvas, f"{size:.2f}", (x - 14, top + plot_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.polylines(canvas, [np.array(latency_points, dtype=np.int32)], False, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(count_points, dtype=np.int32)], False, (0, 180, 80), 2, cv2.LINE_AA)
    for point in latency_points:
        cv2.circle(canvas, point, 5, (0, 120, 255), -1, cv2.LINE_AA)
    for point in count_points:
        cv2.circle(canvas, point, 5, (0, 180, 80), -1, cv2.LINE_AA)
    cv2.putText(canvas, "orange: preprocess latency", (left, top - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "green: pillar count", (left + 320, top - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 80), 2, cv2.LINE_AA)
    return canvas


def compose_grid(images: list[np.ndarray], columns: int = 3, label_lines: list[str] | None = None) -> np.ndarray:
    if not images:
        return placeholder_image(["No images to compose"])
    h = max(image.shape[0] for image in images)
    w = max(image.shape[1] for image in images)
    rows = (len(images) + columns - 1) // columns
    grid = np.full((rows * h, columns * w, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        resized = cv2.resize(image, (w, h))
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = resized
    if label_lines:
        for index, line in enumerate(label_lines[:6]):
            cv2.putText(grid, line, (20, 30 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 220), 2, cv2.LINE_AA)
    return grid

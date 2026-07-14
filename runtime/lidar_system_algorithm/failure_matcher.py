from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .dbscan_baseline import DetectionBox
from .transforms import bev_corners


SUPPORTED_CLASSES = ("Car", "Pedestrian", "Cyclist")
DEFAULT_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
MIN_HEIGHT_BY_DIFFICULTY = {"easy": 40.0, "moderate": 25.0, "hard": 25.0}
MAX_OCCLUSION_BY_DIFFICULTY = {"easy": 0, "moderate": 1, "hard": 2}
MAX_TRUNCATION_BY_DIFFICULTY = {"easy": 0.15, "moderate": 0.3, "hard": 0.5}


@dataclass
class KittiObject:
    class_name: str
    truncation: float
    occlusion: int
    alpha: float
    bbox: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_camera_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None
    frame_id: str

    @property
    def height_2d(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def length(self) -> float:
        return float(self.dimensions_hwl[0])

    @property
    def width(self) -> float:
        return float(self.dimensions_hwl[2])

    @property
    def height(self) -> float:
        return float(self.dimensions_hwl[1])

    @property
    def distance_m(self) -> float:
        x, _, z = self.location_camera_xyz
        return float(math.sqrt(x * x + z * z))


def parse_kitti_object_line(line: str, frame_id: str, is_prediction: bool) -> KittiObject | None:
    parts = line.strip().split()
    if len(parts) < 15:
        return None
    class_name = parts[0]
    if class_name not in SUPPORTED_CLASSES:
        return None
    score = float(parts[15]) if is_prediction and len(parts) >= 16 else None
    hwl = [float(value) for value in parts[8:11]]
    # KITTI txt stores h,w,l. OpenPCDet in-memory annos use l,h,w, so we reorder here
    # to make the analysis matcher geometry consistent with the local official evaluator.
    dimensions_lhw = (hwl[2], hwl[0], hwl[1])
    return KittiObject(
        class_name=class_name,
        truncation=float(parts[1]),
        occlusion=int(float(parts[2])),
        alpha=float(parts[3]),
        bbox=(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
        dimensions_hwl=dimensions_lhw,
        location_camera_xyz=(float(parts[11]), float(parts[12]), float(parts[13])),
        rotation_y=float(parts[14]),
        score=score,
        frame_id=frame_id,
    )


def read_kitti_objects(path: Path, is_prediction: bool) -> list[KittiObject]:
    if not path.exists():
        return []
    objects: list[KittiObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_kitti_object_line(line, frame_id=path.stem, is_prediction=is_prediction)
        if parsed is not None:
            objects.append(parsed)
    return objects


def difficulty_of(obj: KittiObject) -> str:
    if obj.height_2d >= 40.0 and obj.occlusion <= 0 and obj.truncation <= 0.15:
        return "easy"
    if obj.height_2d >= 25.0 and obj.occlusion <= 1 and obj.truncation <= 0.3:
        return "moderate"
    if obj.height_2d >= 25.0 and obj.occlusion <= 2 and obj.truncation <= 0.5:
        return "hard"
    return "unknown"


def official_eval_filter_status(obj: KittiObject, target_class: str, difficulty: str) -> int:
    obj_class = obj.class_name.lower()
    target = target_class.lower()
    valid_class = -1
    if obj_class == target:
        valid_class = 1
    elif target == "pedestrian" and obj_class == "person_sitting":
        valid_class = 0
    elif target == "car" and obj_class == "van":
        valid_class = 0
    else:
        valid_class = -1

    max_occ = MAX_OCCLUSION_BY_DIFFICULTY[difficulty]
    max_trunc = MAX_TRUNCATION_BY_DIFFICULTY[difficulty]
    min_height = MIN_HEIGHT_BY_DIFFICULTY[difficulty]
    ignore = obj.occlusion > max_occ or obj.truncation > max_trunc or obj.height_2d <= min_height
    if valid_class == 1 and not ignore:
        return 0
    if valid_class == 0 or (ignore and valid_class == 1):
        return 1
    return -1


def distance_bin(distance_m: float) -> str:
    if distance_m < 20.0:
        return "0-20m"
    if distance_m < 40.0:
        return "20-40m"
    if distance_m < 60.0:
        return "40-60m"
    return "60m+"


def score_bin(score: float | None) -> str:
    if score is None:
        return "unscored"
    if score < 0.2:
        return "0.0-0.2"
    if score < 0.4:
        return "0.2-0.4"
    if score < 0.6:
        return "0.4-0.6"
    if score < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def size_bin(obj: KittiObject) -> str:
    area = obj.length * obj.width
    if area < 1.5:
        return "small"
    if area < 5.0:
        return "medium"
    return "large"


def bev_polygon_camera(obj: KittiObject) -> np.ndarray:
    x, _, z = obj.location_camera_xyz
    corners = bev_corners(center_x=float(z), center_y=float(x), dx=float(obj.length), dy=float(obj.width), heading=float(obj.rotation_y))
    return corners.astype(np.float64)


def bev_iou(lhs: KittiObject, rhs: KittiObject) -> float:
    lhs_poly = bev_polygon_camera(lhs).astype(np.float32)
    rhs_poly = bev_polygon_camera(rhs).astype(np.float32)
    inter_area, inter_poly = cv2.intersectConvexConvex(lhs_poly, rhs_poly)
    if inter_area <= 0.0:
        return 0.0
    lhs_area = abs(cv2.contourArea(lhs_poly))
    rhs_area = abs(cv2.contourArea(rhs_poly))
    union = lhs_area + rhs_area - float(inter_area)
    return float(inter_area / union) if union > 0 else 0.0


def greedy_match_objects(
    gt_objects: list[KittiObject],
    pred_objects: list[KittiObject],
    iou_thresholds: dict[str, float] | None = None,
) -> tuple[list[dict], list[KittiObject], list[KittiObject]]:
    thresholds = dict(DEFAULT_IOU_THRESHOLDS)
    if iou_thresholds:
        thresholds.update({key: float(value) for key, value in iou_thresholds.items()})
    remaining_gt = {index: obj for index, obj in enumerate(gt_objects)}
    matches: list[dict] = []
    unmatched_preds: list[KittiObject] = []
    sorted_preds = sorted(pred_objects, key=lambda obj: (obj.score if obj.score is not None else -1.0), reverse=True)
    for pred in sorted_preds:
        best_idx = None
        best_iou = -1.0
        for gt_index, gt in remaining_gt.items():
            if gt.class_name != pred.class_name:
                continue
            iou = bev_iou(gt, pred)
            if iou >= thresholds.get(pred.class_name, 0.5) and iou > best_iou:
                best_iou = iou
                best_idx = gt_index
        if best_idx is None:
            unmatched_preds.append(pred)
            continue
        gt = remaining_gt.pop(best_idx)
        matches.append({"gt": gt, "pred": pred, "bev_iou": best_iou})
    unmatched_gt = list(remaining_gt.values())
    return matches, unmatched_preds, unmatched_gt


def as_detection_box(obj: KittiObject, cluster_id: int, source: str) -> DetectionBox:
    x, y, z = obj.location_camera_xyz
    center = np.array([z, x, y], dtype=np.float64)
    size = np.array([obj.length, obj.width, obj.height], dtype=np.float64)
    min_xyz = center - size / 2.0
    max_xyz = center + size / 2.0
    return DetectionBox(
        cluster_id=cluster_id,
        center_xyz=center,
        size_xyz=size,
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        point_count=0,
        distance_m=obj.distance_m,
        azimuth_deg=float(np.degrees(np.arctan2(center[1], center[0]))),
        score=obj.score,
        class_name=obj.class_name,
        yaw=float(obj.rotation_y),
        source=source,
    )

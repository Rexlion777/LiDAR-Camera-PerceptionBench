from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .report_schema import get_project_root


@dataclass
class KittiFrameAssets:
    frame_id: str
    lidar_path: Path
    calib_path: Path | None
    image_path: Path | None
    label_path: Path | None
    dataset_style: str


def locate_default_kitti_root() -> Path:
    project_root = get_project_root()
    candidates = [
        project_root / "data" / "kitti_object_raw" / "extracted",
        project_root / "data" / "kitti_selected_samples",
        project_root / "data" / "kitti_sample",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _dataset_bases(kitti_root: Path) -> Iterable[tuple[str, Path]]:
    root = kitti_root
    if (root / "training" / "velodyne").exists():
        yield "kitti_training", root / "training"
    if (root / "extracted" / "training" / "velodyne").exists():
        yield "kitti_training", root / "extracted" / "training"
    if (root / "velodyne").exists():
        yield "flat_frame_dirs", root
    if (root / "latest" / "sample.bin").exists():
        yield "selected_latest", root / "latest"
    if (root / "sample.bin").exists():
        yield "single_sample", root
    for child in sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []:
        if (child / "velodyne").exists():
            yield "selected_frame_dir", child


def list_frame_ids(kitti_root: Path, limit: int | None = None) -> list[str]:
    frame_ids: list[str] = []
    seen: set[str] = set()
    for style, base in _dataset_bases(kitti_root):
        if style in {"kitti_training", "flat_frame_dirs"}:
            lidar_dir = base / "velodyne"
            for bin_path in sorted(lidar_dir.glob("*.bin")):
                frame_id = bin_path.stem
                if frame_id not in seen:
                    seen.add(frame_id)
                    frame_ids.append(frame_id)
        elif style in {"selected_latest", "single_sample"}:
            if "sample" not in seen:
                seen.add("sample")
                frame_ids.append("sample")
        elif style == "selected_frame_dir":
            frame_id = base.name
            if frame_id not in seen and (base / "velodyne" / f"{frame_id}.bin").exists():
                seen.add(frame_id)
                frame_ids.append(frame_id)
        if limit is not None and len(frame_ids) >= limit:
            break
    return frame_ids[:limit] if limit is not None else frame_ids


def resolve_frame_assets(kitti_root: Path, frame_id: str) -> KittiFrameAssets:
    for style, base in _dataset_bases(kitti_root):
        if style in {"kitti_training", "flat_frame_dirs"}:
            lidar_path = base / "velodyne" / f"{frame_id}.bin"
            if lidar_path.exists():
                return KittiFrameAssets(
                    frame_id=frame_id,
                    lidar_path=lidar_path,
                    calib_path=(base / "calib" / f"{frame_id}.txt") if (base / "calib" / f"{frame_id}.txt").exists() else None,
                    image_path=(base / "image_2" / f"{frame_id}.png") if (base / "image_2" / f"{frame_id}.png").exists() else None,
                    label_path=(base / "label_2" / f"{frame_id}.txt") if (base / "label_2" / f"{frame_id}.txt").exists() else None,
                    dataset_style=style,
                )
        elif style in {"selected_latest", "single_sample"} and frame_id in {"sample", "latest", "000000"}:
            return KittiFrameAssets(
                frame_id="sample",
                lidar_path=base / "sample.bin",
                calib_path=(base / "sample_calib.txt") if (base / "sample_calib.txt").exists() else None,
                image_path=(base / "sample_image.png") if (base / "sample_image.png").exists() else None,
                label_path=(base / "sample_label.txt") if (base / "sample_label.txt").exists() else None,
                dataset_style=style,
            )
        elif style == "selected_frame_dir" and frame_id == base.name:
            return KittiFrameAssets(
                frame_id=frame_id,
                lidar_path=base / "velodyne" / f"{frame_id}.bin",
                calib_path=(base / "calib" / f"{frame_id}.txt") if (base / "calib" / f"{frame_id}.txt").exists() else None,
                image_path=(base / "image_2" / f"{frame_id}.png") if (base / "image_2" / f"{frame_id}.png").exists() else None,
                label_path=(base / "label_2" / f"{frame_id}.txt") if (base / "label_2" / f"{frame_id}.txt").exists() else None,
                dataset_style=style,
            )
    raise FileNotFoundError(f"Unable to locate KITTI frame '{frame_id}' under: {kitti_root}")


def read_kitti_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size == 0 or raw.size % 4 != 0:
        raise ValueError(f"Invalid KITTI point cloud file: {path}")
    return raw.reshape(-1, 4)


def read_image_bgr(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def save_image_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode image for: {path}")
    encoded.tofile(str(path))

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "P2": (3, 4),
    "R0_rect": (3, 3),
    "Tr_velo_to_cam": (3, 4),
}


@dataclass
class Calibration:
    p2: np.ndarray
    r0_rect: np.ndarray
    tr_velo_to_cam: np.ndarray

    @property
    def r0_rect_4x4(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.r0_rect
        return matrix

    @property
    def tr_velo_to_cam_4x4(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = self.tr_velo_to_cam
        return matrix

    def as_dict(self) -> dict:
        return {
            "P2": self.p2.tolist(),
            "R0_rect": self.r0_rect.tolist(),
            "Tr_velo_to_cam": self.tr_velo_to_cam.tolist(),
        }


def parse_calibration_file(calib_path: Path) -> Calibration:
    matrices: dict[str, np.ndarray] = {}
    for raw_line in calib_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, values_str = line.split(":", 1)
        key = key.strip()
        if key not in REQUIRED_KEYS:
            continue
        expected_shape = REQUIRED_KEYS[key]
        values = values_str.strip().split()
        expected_size = expected_shape[0] * expected_shape[1]
        if len(values) != expected_size:
            raise ValueError(f"Calibration matrix {key} expected {expected_size} values, got {len(values)}")
        matrices[key] = np.asarray([float(value) for value in values], dtype=np.float64).reshape(expected_shape)
    missing = [key for key in REQUIRED_KEYS if key not in matrices]
    if missing:
        raise ValueError(f"Missing calibration fields: {', '.join(missing)}")
    return Calibration(
        p2=matrices["P2"],
        r0_rect=matrices["R0_rect"],
        tr_velo_to_cam=matrices["Tr_velo_to_cam"],
    )

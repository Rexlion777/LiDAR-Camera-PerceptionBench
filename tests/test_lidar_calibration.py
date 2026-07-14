from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from runtime.lidar_system_algorithm.calibration import parse_calibration_file
from runtime.lidar_system_algorithm.transforms import project_lidar_to_image


CALIB_TEXT = """P2: 7 0 6 0 0 7 2 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
"""


class TestLidarCalibration(unittest.TestCase):
    def test_parse_calibration_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calib_path = Path(temp_dir) / "sample.txt"
            calib_path.write_text(CALIB_TEXT, encoding="utf-8")
            calibration = parse_calibration_file(calib_path)
        self.assertEqual(calibration.p2.shape, (3, 4))
        self.assertEqual(calibration.r0_rect.shape, (3, 3))
        self.assertEqual(calibration.tr_velo_to_cam.shape, (3, 4))

    def test_projection_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            calib_path = Path(temp_dir) / "sample.txt"
            calib_path.write_text(CALIB_TEXT, encoding="utf-8")
            calibration = parse_calibration_file(calib_path)
        points = np.array([[5.0, 0.0, 10.0], [10.0, 1.0, 12.0]], dtype=np.float64)
        rectified, uv, valid = project_lidar_to_image(points, calibration)
        self.assertEqual(rectified.shape, (2, 3))
        self.assertEqual(uv.shape, (2, 2))
        self.assertEqual(valid.shape, (2,))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from runtime.lidar_system_algorithm.calibration import Calibration
from runtime.lidar_system_algorithm.transforms import lidar_to_rectified_camera, project_rectified_to_image, to_homogeneous


class TestLidarTransforms(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = Calibration(
            p2=np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64),
            r0_rect=np.eye(3, dtype=np.float64),
            tr_velo_to_cam=np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]], dtype=np.float64),
        )

    def test_to_homogeneous(self) -> None:
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = to_homogeneous(points)
        np.testing.assert_allclose(result, np.array([[1.0, 2.0, 3.0, 1.0]], dtype=np.float64))

    def test_lidar_to_camera_transform(self) -> None:
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        rectified = lidar_to_rectified_camera(points, self.calibration)
        np.testing.assert_allclose(rectified, np.array([[2.0, 4.0, 6.0]], dtype=np.float64))

    def test_projection_shape(self) -> None:
        points_rect = np.array([[2.0, 4.0, 6.0]], dtype=np.float64)
        uv, valid = project_rectified_to_image(points_rect, self.calibration)
        self.assertEqual(uv.shape, (1, 2))
        self.assertEqual(valid.shape, (1,))
        self.assertTrue(bool(valid[0]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from runtime.lidar_system_algorithm.dbscan_baseline import build_dbscan_baseline


class TestLidarDbscanBaseline(unittest.TestCase):
    def test_synthetic_clusters_generate_boxes(self) -> None:
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=[10.0, 0.0, 0.0], scale=[0.2, 0.2, 0.1], size=(40, 3))
        cluster_b = rng.normal(loc=[20.0, 5.0, 0.1], scale=[0.2, 0.2, 0.1], size=(45, 3))
        points = np.vstack([cluster_a, cluster_b]).astype(np.float64)
        result = build_dbscan_baseline(
            points_xyz=points,
            roi=((0.0, 30.0), (-10.0, 10.0), (-1.0, 1.0)),
            ground_z_threshold=None,
            eps=0.8,
            min_points=5,
            min_cluster_points=10,
        )
        self.assertGreaterEqual(len(result["boxes"]), 2)
        for box in result["boxes"]:
            self.assertGreater(box.size_xyz[0], 0.0)
            self.assertGreater(box.size_xyz[1], 0.0)


if __name__ == "__main__":
    unittest.main()

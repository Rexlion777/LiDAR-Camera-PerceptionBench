import unittest

import numpy as np

from runtime.lidar_system_algorithm.tracking import Detection, OptimizedMultiObjectTracker


class OptimizedTrackingTest(unittest.TestCase):
    def _det(self, frame_id, x, y):
        return Detection(
            frame_id=frame_id,
            center_xyz=np.array([x, y, 0.0], dtype=float),
            size_xyz=np.array([4.0, 1.8, 1.5], dtype=float),
            class_name="Car",
        )

    def test_keeps_track_id_and_reports_gating(self):
        tracker = OptimizedMultiObjectTracker(distance_threshold=3.0, max_age=2, min_hits=1)
        tracks0, stats0 = tracker.update_with_stats([self._det("0", 0, 0), self._det("0", 10, 0)], "0")
        tracks1, stats1 = tracker.update_with_stats([self._det("1", 0.5, 0), self._det("1", 10.2, 0.1)], "1")
        self.assertEqual([t.track_id for t in tracks0], [1, 2])
        self.assertEqual([t.track_id for t in tracks1], [1, 2])
        self.assertGreaterEqual(stats1["gated_pair_count"], 2)
        self.assertLessEqual(stats1["association_matrix_size"], 4)


if __name__ == "__main__":
    unittest.main()

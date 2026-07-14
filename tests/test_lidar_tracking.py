from __future__ import annotations

import unittest

import numpy as np

from runtime.lidar_system_algorithm.tracking import Detection, MultiObjectTracker


class TestLidarTracking(unittest.TestCase):
    def test_track_id_persists_across_two_frames(self) -> None:
        tracker = MultiObjectTracker(distance_threshold=3.0, max_age=2, min_hits=1)
        detections_frame1 = [
            Detection(frame_id="000000", center_xyz=np.array([10.0, 0.0, 0.0]), size_xyz=np.array([4.0, 2.0, 1.5]))
        ]
        tracks_frame1 = tracker.update(detections_frame1, frame_id="000000")
        detections_frame2 = [
            Detection(frame_id="000001", center_xyz=np.array([10.8, 0.2, 0.0]), size_xyz=np.array([4.0, 2.0, 1.5]))
        ]
        tracks_frame2 = tracker.update(detections_frame2, frame_id="000001")
        self.assertEqual(len(tracks_frame1), 1)
        self.assertEqual(len(tracks_frame2), 1)
        self.assertEqual(tracks_frame1[0].track_id, tracks_frame2[0].track_id)

    def test_unmatched_track_aging(self) -> None:
        tracker = MultiObjectTracker(distance_threshold=2.0, max_age=1, min_hits=1)
        tracker.update([Detection(frame_id="0", center_xyz=np.array([0.0, 0.0, 0.0]), size_xyz=np.array([1.0, 1.0, 1.0]))], frame_id="0")
        tracker.update([], frame_id="1")
        tracker.update([], frame_id="2")
        self.assertEqual(len(tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()

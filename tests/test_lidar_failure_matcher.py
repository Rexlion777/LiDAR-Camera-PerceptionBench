import unittest

from runtime.lidar_system_algorithm.failure_matcher import KittiObject, distance_bin, greedy_match_objects, official_eval_filter_status


def make_obj(class_name: str, x: float, z: float, score=None) -> KittiObject:
    return KittiObject(
        class_name=class_name,
        truncation=0.0,
        occlusion=0,
        alpha=0.0,
        bbox=(0.0, 0.0, 50.0, 60.0),
        dimensions_hwl=(1.5, 1.8, 4.0),
        location_camera_xyz=(x, 1.5, z),
        rotation_y=0.0,
        score=score,
        frame_id="000001",
    )


class FailureMatcherTest(unittest.TestCase):
    def test_tp_fp_fn(self):
        gt = [make_obj("Car", 0.0, 20.0), make_obj("Pedestrian", 5.0, 15.0)]
        preds = [make_obj("Car", 0.0, 20.0, 0.9), make_obj("Cyclist", 20.0, 20.0, 0.6)]
        matches, unmatched_preds, unmatched_gt = greedy_match_objects(gt, preds)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(unmatched_preds), 1)
        self.assertEqual(len(unmatched_gt), 1)

    def test_range_bin(self):
        self.assertEqual(distance_bin(10.0), "0-20m")
        self.assertEqual(distance_bin(30.0), "20-40m")
        self.assertEqual(distance_bin(50.0), "40-60m")
        self.assertEqual(distance_bin(80.0), "60m+")

    def test_official_filter_status(self):
        car = make_obj("Car", 0.0, 20.0)
        self.assertEqual(official_eval_filter_status(car, "Car", "moderate"), 0)
        short = KittiObject(
            class_name="Car",
            truncation=0.0,
            occlusion=0,
            alpha=0.0,
            bbox=(0.0, 0.0, 20.0, 10.0),
            dimensions_hwl=(4.0, 1.5, 1.8),
            location_camera_xyz=(0.0, 1.5, 20.0),
            rotation_y=0.0,
            score=None,
            frame_id="000001",
        )
        self.assertEqual(official_eval_filter_status(short, "Car", "moderate"), 1)


if __name__ == "__main__":
    unittest.main()

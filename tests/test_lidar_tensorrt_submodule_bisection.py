import json
import unittest
from pathlib import Path

from runtime.lidar_system_algorithm.tensorrt_debug_utils import pad_prepared_inputs
import numpy as np


class TensorRTSubmoduleBisectionTest(unittest.TestCase):
    def test_unique_dummy_padding_supports_3d_coords(self):
        prepared = {
            "voxels": np.zeros((2, 32, 4), dtype=np.float32),
            "voxel_num_points": np.ones((2,), dtype=np.int32),
            "voxel_coords": np.asarray([[0, 1, 1], [0, 1, 2]], dtype=np.int32),
        }
        padded, stats = pad_prepared_inputs(prepared, 4, "unique_dummy_coord_padding")
        self.assertEqual(padded["voxel_coords"].shape[0], 4)
        self.assertEqual(stats["padded_pillar_count"], 2)

    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_submodule_bisection_report.json")
        if not path.exists():
            self.skipTest("submodule bisection report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertTrue(payload.get("judgements"))


if __name__ == "__main__":
    unittest.main()

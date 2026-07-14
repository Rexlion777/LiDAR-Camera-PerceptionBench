import json
import unittest
from pathlib import Path


class TensorRTScatterPaddingExperimentTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_scatter_padding_experiment.json")
        if not path.exists():
            self.skipTest("scatter padding experiment report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        strategies = {row.get("strategy") for row in payload.get("results", [])}
        self.assertIn("repeat_first_valid", strategies)
        self.assertIn("unique_dummy_coord_padding", strategies)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path


class TensorRTRealSampleWrapperTest(unittest.TestCase):
    def test_wrapper_report_schema_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_real_sample_wrapper.json")
        if not path.exists():
            self.skipTest("TensorRT real-sample wrapper report not generated in this environment")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(payload.get("status"), {"completed", "skipped", "failed"})
        self.assertIn("scope", payload)
        if payload.get("status") == "completed":
            self.assertIn("mean_trt_core_ms", payload)
            self.assertIn("limitations", payload)


if __name__ == "__main__":
    unittest.main()

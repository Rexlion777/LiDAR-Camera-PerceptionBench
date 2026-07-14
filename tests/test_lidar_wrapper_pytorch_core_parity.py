import json
import unittest
from pathlib import Path


class WrapperPyTorchCoreParityTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/wrapper_pytorch_core_parity_report.json")
        if not path.exists():
            self.skipTest("wrapper parity report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("ap_summary", payload)
        self.assertIn("summary", payload)


if __name__ == "__main__":
    unittest.main()

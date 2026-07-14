import json
import unittest
from pathlib import Path


class TensorRTDecodeAlignmentAuditTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_decode_alignment_audit.json")
        if not path.exists():
            self.skipTest("decode alignment audit not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("checks", payload)
        self.assertIn("suspected_findings", payload)


if __name__ == "__main__":
    unittest.main()

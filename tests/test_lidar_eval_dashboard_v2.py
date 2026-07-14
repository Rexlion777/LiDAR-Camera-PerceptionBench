import json
import unittest
from pathlib import Path


class EvalDashboardV2Test(unittest.TestCase):
    def test_dashboard_v2_schema_if_present(self):
        path = Path("reports/lidar_system_algorithm/eval_dashboard.json")
        if not path.exists():
            self.skipTest("eval dashboard not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertIn("version", payload)
        self.assertIn("failure_analysis_summary", payload)
        self.assertIn("tensorrt_bucketed_summary", payload)
        note = payload.get("failure_case_note", "")
        self.assertIn("official", note.lower())


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from pathlib import Path


class WrapperBatchDictAuditTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/wrapper_pytorch_core_batch_dict_audit.json")
        if not path.exists():
            self.skipTest("wrapper batch_dict audit report not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        findings = payload.get("global_findings", {})
        self.assertIn("wrapper_uses_native_post_processing", findings)
        self.assertIn("suspected_contract_issue", findings)


if __name__ == "__main__":
    unittest.main()

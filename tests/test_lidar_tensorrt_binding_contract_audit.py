import json
import unittest
from pathlib import Path


class TensorRTBindingContractAuditTest(unittest.TestCase):
    def test_report_if_present(self):
        path = Path("reports/lidar_system_algorithm/tensorrt_binding_contract_audit.json")
        if not path.exists():
            self.skipTest("binding contract audit not generated")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("status"), "completed")
        self.assertTrue(payload.get("contract_checks", {}).get("outputs_fetched_by_name"))
        mapping = payload.get("mapping_table", [])
        self.assertTrue(any(row.get("pytorch_key") == "batch_cls_preds" for row in mapping))


if __name__ == "__main__":
    unittest.main()

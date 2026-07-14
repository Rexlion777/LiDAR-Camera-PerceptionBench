from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestCudaLibdeviceAudit(unittest.TestCase):
    def test_audit_outputs_exist(self) -> None:
        audit_json = REPORT_ROOT / "cuda_libdevice_audit.json"
        audit_md = REPORT_ROOT / "cuda_libdevice_audit.md"
        self.assertTrue(audit_json.exists())
        self.assertTrue(audit_md.exists())
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        for key in ("libdevice_candidates", "libnvvm_candidates", "recommended_env", "manual_fix_suggestions"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["libdevice_candidates"], list)
        self.assertIsInstance(payload["libnvvm_candidates"], list)


if __name__ == "__main__":
    unittest.main()

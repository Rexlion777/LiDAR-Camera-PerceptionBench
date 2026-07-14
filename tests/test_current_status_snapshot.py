from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestCurrentStatusSnapshot(unittest.TestCase):
    def test_snapshot_files_exist(self) -> None:
        self.assertTrue((REPORT_ROOT / "current_status_snapshot.md").exists())
        self.assertTrue((REPORT_ROOT / "current_status_snapshot.json").exists())

    def test_snapshot_has_expected_status_fields(self) -> None:
        payload = json.loads((REPORT_ROOT / "current_status_snapshot.json").read_text(encoding="utf-8"))
        for key in (
            "smoke_train_status",
            "subset_finetune_status",
            "native_tools_test_eval_status",
            "external_wrapper_eval_status",
            "deployment_diagnostics_status",
            "safe_claims",
            "limitations",
            "forbidden_claims",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["native_tools_test_eval_status"], "completed")
        self.assertEqual(payload["subset_finetune_status"]["opencpdet_tools_eval_status"], "completed")


if __name__ == "__main__":
    unittest.main()

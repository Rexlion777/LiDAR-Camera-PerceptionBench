from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestOpenPCDetToolsEvalRetryStatus(unittest.TestCase):
    def test_retry_status_exists_and_has_reason(self) -> None:
        retry_path = REPORT_ROOT / "openpcdet_tools_eval_retry_status.json"
        self.assertTrue(retry_path.exists())
        payload = json.loads(retry_path.read_text(encoding="utf-8"))
        self.assertIn("attempts", payload)
        self.assertGreater(len(payload["attempts"]), 0)
        reasons = {attempt.get("reason") for attempt in payload["attempts"]}
        allowed = {"", "failed_due_missing_libdevice", "failed_due_missing_libnvvm", "failed_due_torch_checkpoint_loading"}
        self.assertTrue(reasons.issubset(allowed) or any(str(reason).startswith("failed_exit_code_") for reason in reasons if reason))

    def test_per_experiment_status_matches_retry(self) -> None:
        for experiment_name in ("smoke_train_20260630_112530", "subset_finetune_20260630_112826"):
            status_path = REPORT_ROOT / experiment_name / "train_status.json"
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            tools_status = payload.get("experiment_status", {}).get("opencpdet_tools_eval_status", "")
            self.assertIn(
                tools_status,
                {"completed", "failed_due_missing_libdevice", "failed_due_missing_libnvvm", "failed_due_torch_checkpoint_loading"},
            )
            if tools_status != "completed":
                self.assertIn("overall_claim_level", payload.get("experiment_status", {}))


if __name__ == "__main__":
    unittest.main()

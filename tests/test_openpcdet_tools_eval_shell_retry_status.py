from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestOpenPCDetToolsEvalShellRetryStatus(unittest.TestCase):
    def test_shell_wrapper_and_outputs_exist(self) -> None:
        self.assertTrue((PROJECT_ROOT / "scripts" / "lidar_system_algorithm" / "run_openpcdet_eval_with_cuda_env.sh").exists())
        self.assertTrue((REPORT_ROOT / "openpcdet_tools_eval_shell_retry.log").exists())
        self.assertTrue((REPORT_ROOT / "openpcdet_tools_eval_shell_retry_status.json").exists())

    def test_status_has_required_fields(self) -> None:
        payload = json.loads((REPORT_ROOT / "openpcdet_tools_eval_shell_retry_status.json").read_text(encoding="utf-8"))
        for key in ("sanity_check_status", "ctypes_load_libnvvm", "tools_eval_status", "attempts"):
            self.assertIn(key, payload)
        if payload.get("tools_eval_status") != "completed":
            self.assertTrue(payload.get("reason"))

    def test_completed_requires_success_log(self) -> None:
        payload = json.loads((REPORT_ROOT / "openpcdet_tools_eval_shell_retry_status.json").read_text(encoding="utf-8"))
        log_text = (REPORT_ROOT / "openpcdet_tools_eval_shell_retry.log").read_text(encoding="utf-8", errors="replace")
        if payload.get("tools_eval_status") == "completed":
            self.assertIn("TOOLS_TEST_COMPLETED", log_text)


if __name__ == "__main__":
    unittest.main()

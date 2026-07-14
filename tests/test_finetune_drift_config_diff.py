from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune" / "diagnose_finetune_drift"


class TestFineTuneDriftConfigDiff(unittest.TestCase):
    def test_config_diff_outputs_exist(self) -> None:
        self.assertTrue((DIAG_ROOT / "eval_config_diff.md").exists())
        self.assertTrue((DIAG_ROOT / "eval_config_diff.json").exists())

    def test_core_eval_config_is_consistent(self) -> None:
        payload = json.loads((DIAG_ROOT / "eval_config_diff.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["overall_consistent"])
        by_field = {item["field"]: item for item in payload["diffs"]}
        self.assertTrue(by_field["MODEL.POST_PROCESSING"]["consistent"])
        self.assertTrue(by_field["MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG"]["consistent"])
        self.assertTrue(by_field["MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG"]["consistent"])


if __name__ == "__main__":
    unittest.main()

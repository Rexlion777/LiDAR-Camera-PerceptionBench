from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestLidarTrainingEnvAudit(unittest.TestCase):
    def test_training_env_audit_exists(self) -> None:
        audit_json = REPORT_ROOT / "training_env_audit.json"
        audit_md = REPORT_ROOT / "training_env_audit.md"
        self.assertTrue(audit_json.exists())
        self.assertTrue(audit_md.exists())
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        for key in (
            "opencpdet_path",
            "kitti_data_root",
            "config_path",
            "pretrained_checkpoint_path",
            "cuda_available",
            "spconv_available",
            "pcdet_ops_available",
            "train_entry_available",
            "test_entry_available",
        ):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestTrainingStatusGranularity(unittest.TestCase):
    def test_train_status_is_granular(self) -> None:
        for experiment_name in ("smoke_train_20260630_112530", "subset_finetune_20260630_112826"):
            status_path = REPORT_ROOT / experiment_name / "train_status.json"
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            experiment_status = payload.get("experiment_status", {})
            self.assertEqual(experiment_status.get("training_status"), "completed")
            self.assertEqual(experiment_status.get("checkpoint_status"), "completed")
            self.assertEqual(experiment_status.get("external_eval_wrapper_status"), "completed")
            self.assertNotEqual(payload.get("status"), "partial")
            self.assertTrue(str(payload.get("pretrained_checkpoint", "")).endswith(".pth"))
            self.assertNotIn(".pt h", str(payload.get("pretrained_checkpoint", "")))

    def test_report_does_not_mark_whole_experiment_partial(self) -> None:
        report_path = REPORT_ROOT / "pointpillars_training_finetune_report.md"
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("Training completed and checkpoints were preserved", text)
        self.assertNotIn("whole experiment partial", text.lower())


if __name__ == "__main__":
    unittest.main()

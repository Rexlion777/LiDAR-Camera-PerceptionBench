from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestExpandedFineTuneOutputs(unittest.TestCase):
    def test_expanded_experiment_outputs_exist(self) -> None:
        exp_dir = REPORT_ROOT / "expanded_finetune_1000_20260630_150229"
        self.assertTrue(exp_dir.exists())
        for name in (
            "run_config.yaml",
            "train_command.txt",
            "eval_command.txt",
            "train.log",
            "eval.log",
            "train_status.json",
            "checkpoint_index.json",
            "metrics_summary.json",
            "loss_curve.csv",
            "eval_ap_summary.csv",
            "safe_claims.md",
            "forbidden_claims.md",
        ):
            self.assertTrue((exp_dir / name).exists(), name)

    def test_expanded_status_is_granular_and_completed(self) -> None:
        exp_dir = REPORT_ROOT / "expanded_finetune_1000_20260630_150229"
        status = json.loads((exp_dir / "train_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["train_sample_count"], 1000)
        self.assertEqual(status["val_sample_count"], 200)
        self.assertEqual(status["epochs"], 3)
        self.assertEqual(status["batch_size"], 2)
        self.assertAlmostEqual(float(status["learning_rate"]), 0.0008, places=7)
        self.assertTrue(status["used_pretrained_checkpoint"])
        exp_status = status["experiment_status"]
        self.assertEqual(exp_status["training_status"], "completed")
        self.assertEqual(exp_status["checkpoint_status"], "completed")
        self.assertEqual(exp_status["external_eval_wrapper_status"], "completed")
        self.assertEqual(exp_status["opencpdet_tools_eval_status"], "completed")
        self.assertEqual(exp_status["train_py_inline_eval_status"], "failed_due_train_py_inline_eval")
        self.assertIn("weights_only", status["partial_reason"])

    def test_checkpoint_path_exists(self) -> None:
        exp_dir = REPORT_ROOT / "expanded_finetune_1000_20260630_150229"
        status = json.loads((exp_dir / "train_status.json").read_text(encoding="utf-8"))
        trained_checkpoint = Path(status["trained_checkpoint"])
        self.assertTrue(trained_checkpoint.exists())


if __name__ == "__main__":
    unittest.main()

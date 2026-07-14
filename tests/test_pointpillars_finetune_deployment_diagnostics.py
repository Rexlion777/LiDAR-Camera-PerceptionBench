from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "lidar_system_algorithm" / "training_finetune"


class TestPointPillarsFinetuneDeploymentDiagnostics(unittest.TestCase):
    def test_diagnostics_exist_and_have_real_counts(self) -> None:
        diagnostics_json = REPORT_ROOT / "finetune_deployment_diagnostics.json"
        diagnostics_md = REPORT_ROOT / "finetune_deployment_diagnostics.md"
        self.assertTrue(diagnostics_json.exists())
        self.assertTrue(diagnostics_md.exists())
        payload = json.loads(diagnostics_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["scope"], "subset-val deployment-style diagnostics")
        self.assertEqual(payload["baseline"]["prediction_health"]["empty_prediction_file_count"], 0)
        self.assertEqual(payload["fine_tuned"]["prediction_health"]["empty_prediction_file_count"], 0)
        self.assertGreater(payload["baseline"]["prediction_health"]["total_box_count"], 0)
        self.assertGreater(payload["fine_tuned"]["prediction_health"]["total_box_count"], 0)


if __name__ == "__main__":
    unittest.main()

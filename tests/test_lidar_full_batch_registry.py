import csv
import json
import unittest
from pathlib import Path


class FullBatchRegistryTest(unittest.TestCase):
    def test_run_registry_exists_and_has_expected_fields(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance/run_registry")
        csv_path = root / "all_settings_registry.csv"
        json_path = root / "all_settings_registry.json"
        last_run = root / "full_batch_last_run.json"
        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())
        self.assertTrue(last_run.exists())

        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual(payload.get("row_count"), len(rows))
        for field in [
            "setting_id",
            "perturbation_type",
            "perturbation_value",
            "mode",
            "status",
            "frame_count",
            "sampling_mode",
            "command",
        ]:
            self.assertIn(field, rows[0])
        modes = {row["mode"] for row in rows}
        self.assertIn("quick_dense", modes)
        self.assertIn("selected_1000", modes)
        self.assertIn("proxy_extended", modes)


if __name__ == "__main__":
    unittest.main()

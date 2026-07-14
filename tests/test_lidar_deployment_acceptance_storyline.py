import json
import re
import unittest
from pathlib import Path


class DeploymentAcceptanceStorylineTest(unittest.TestCase):
    def test_storyline_and_safe_claims(self):
        root = Path("reports/lidar_system_algorithm/deployment_acceptance")
        story = json.loads((root / "deployment_acceptance_ppt_storyline.json").read_text(encoding="utf-8"))
        report = json.loads((root / "deployment_acceptance_final_report.json").read_text(encoding="utf-8"))
        self.assertEqual(story.get("language"), "zh-CN")
        self.assertEqual(len(story.get("slides", [])), 5)
        cjk = re.compile(r"[\u4e00-\u9fff]")
        for slide in story["slides"]:
            self.assertIn("forbidden_claims", slide)
            self.assertTrue(slide.get("title"))
            self.assertTrue(cjk.search(slide.get("title", "")))
            self.assertTrue(slide.get("takeaway"))
            self.assertGreaterEqual(len(slide.get("bullets", [])), 3)
            self.assertIn("full TensorRT detector", " ".join(slide.get("forbidden_claims", [])))
        forbidden = " ".join(report.get("forbidden_claims", []))
        self.assertIn("full TensorRT detector", forbidden)
        self.assertIn("new 3D detection model", forbidden)
        self.assertIn("full-val if only 1000-frame slice", forbidden)
        self.assertTrue(report.get("dense_diagnostics_needed"))
        self.assertIn("why_dense_diagnostics_are_needed", report)
        self.assertEqual(report.get("selected_1000_scope", {}).get("eval_scope"), "1000-frame-slice")


if __name__ == "__main__":
    unittest.main()

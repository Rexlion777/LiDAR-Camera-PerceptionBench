import unittest
from pathlib import Path

from PIL import Image


FIGURE_NAMES = [f"{index:02d}_" for index in range(1, 43)]
SLIDES = [
    "slide1_application_and_acceptance_chain",
    "slide2_pointcloud_and_range_degradation",
    "slide3_distance_noise_and_projection_proxy",
    "slide4_time_and_postprocess_diagnostics",
    "slide5_deployment_precision_and_health_monitoring",
]


class HighResFiguresTest(unittest.TestCase):
    def test_dense_figure_resolution_and_triplets(self):
        fig_dir = Path("projects/lidar_system_algorithm/figures/deployment_acceptance_dense")
        ppt_dir = Path("projects/lidar_system_algorithm/figures/deployment_acceptance_dense_ppt_panels")
        contact_sheet = fig_dir / "deployment_acceptance_dense_contact_sheet.png"
        self.assertTrue(contact_sheet.exists())
        with Image.open(contact_sheet) as image:
            width, height = image.size
        self.assertGreaterEqual(width, 8000)
        self.assertGreater(height, 0)

        for stem in FIGURE_NAMES:
            matches = [path for path in fig_dir.glob(f"{stem}*.png") if "contact_sheet" not in path.name]
            self.assertTrue(matches, stem)
            png = matches[0]
            self.assertTrue(png.with_suffix(".svg").exists(), png.name)
            self.assertTrue(png.with_suffix(".pdf").exists(), png.name)
            with Image.open(png) as image:
                w, h = image.size
            self.assertGreaterEqual(w, 2400)
            self.assertGreaterEqual(h, 1800)

        for slide in SLIDES:
            png = ppt_dir / f"{slide}.png"
            self.assertTrue(png.exists(), slide)
            self.assertTrue((ppt_dir / f"{slide}.svg").exists(), slide)
            self.assertTrue((ppt_dir / f"{slide}.pdf").exists(), slide)
            with Image.open(png) as image:
                w, h = image.size
            self.assertGreaterEqual(w, 3840)
            self.assertGreaterEqual(h, 2160)


if __name__ == "__main__":
    unittest.main()

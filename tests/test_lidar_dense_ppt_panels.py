import unittest
from pathlib import Path

from PIL import Image


SLIDES = [
    "slide1_application_and_acceptance_chain",
    "slide2_pointcloud_and_range_degradation",
    "slide3_distance_noise_and_projection_proxy",
    "slide4_time_and_postprocess_diagnostics",
    "slide5_deployment_precision_and_health_monitoring",
]


class DensePptPanelsTest(unittest.TestCase):
    def test_dense_ppt_panels_exist(self):
        ppt_dir = Path("projects/lidar_system_algorithm/figures/deployment_acceptance_dense_ppt_panels")
        for slide in SLIDES:
            png = ppt_dir / f"{slide}.png"
            svg = ppt_dir / f"{slide}.svg"
            pdf = ppt_dir / f"{slide}.pdf"
            self.assertTrue(png.exists(), slide)
            self.assertTrue(svg.exists(), slide)
            self.assertTrue(pdf.exists(), slide)
            with Image.open(png) as image:
                w, h = image.size
            self.assertGreaterEqual(w, 3840)
            self.assertGreaterEqual(h, 2160)


if __name__ == "__main__":
    unittest.main()

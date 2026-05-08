import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from display_window import DisplayWindow
from file_manager import FileManager


class TestDisplayWindowHistoricTiles(unittest.TestCase):
    def _build_display(self):
        with patch("display_window.get_db_connection", return_value=MagicMock()):
            display = DisplayWindow(
                width=800,
                height=600,
                refresh_interval=0,
                file_manager=FileManager(),
            )
        display.show = MagicMock(return_value=True)
        display.get_result_for_image = MagicMock(return_value="OK")
        display._get_piece_result_counts = MagicMock(
            return_value={"OK": 0, "NOK": 0, "FOK": 0, "FNOK": 0}
        )
        return display

    def test_overlay_parser_accepts_remote_bbox_dict(self):
        display = self._build_display()

        points = display._points_from_overlay_coordinates(
            {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
            "bbox",
        )

        self.assertEqual(points, [(10.0, 20.0), (100.0, 20.0), (100.0, 120.0), (10.0, 120.0)])

    def test_overlay_parser_accepts_remote_polygon_points(self):
        display = self._build_display()

        points = display._points_from_overlay_coordinates(
            {"points": [{"x": 10, "y": 20}, {"x": 100, "y": 20}, {"x": 50, "y": 120}]},
            "polygon",
        )

        self.assertEqual(points, [(10.0, 20.0), (100.0, 20.0), (50.0, 120.0)])

    def test_overlay_scaling_uses_remote_image_dimensions(self):
        display = self._build_display()

        scaled = display._scale_overlay_points(
            [(10.0, 20.0), (100.0, 120.0)],
            {"image_width": 200, "image_height": 240},
            target_w=100,
            target_h=120,
            fallback_source_w=360,
            fallback_source_h=360,
        )

        self.assertEqual(scaled, [(5, 10), (50, 60)])

    def test_show_image_grid_draws_historic_loading_tile(self):
        display = self._build_display()
        display.historic_mode = True
        display.historic_images = [["118610000000000000001_Cam1_Side1_OK.png"]]

        result = display.show_image_grid(
            [
                {
                    "img_name": "118610000000000000001_Cam1_Side1_OK.png",
                    "status": "loading",
                    "source": "db_coordinates+historic",
                }
            ],
            cols=2,
            rows=1,
            img_size=100,
            padding=10,
        )

        self.assertTrue(result)
        self.assertIsInstance(display.image, np.ndarray)
        self.assertLess(int(display.image[250:350, 295:395].mean()), 250)

    def test_show_image_grid_draws_filename_status_badge_in_historic_mode(self):
        display = self._build_display()
        display.historic_mode = True
        display.historic_images = [["118610000000000000001_Cam1_Side1_NOK.png"]]
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = display.show_image_grid(
            [
                {
                    "img_name": "118610000000000000001_Cam1_Side1_NOK.png",
                    "status": "ready",
                    "source": "db_coordinates+historic",
                    "prepared_image": image,
                }
            ],
            cols=2,
            rows=1,
            img_size=100,
            padding=10,
        )

        self.assertTrue(result)
        badge_region = display.image[258:285, 365:395]
        self.assertGreater(int(badge_region[:, :, 2].mean()), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from file_manager import FileManager
from main_controller import ControllerConfig, MainController


class TestHistoricConfidenceFilter(unittest.TestCase):
    def _build_display(self, db=None):
        display = MagicMock()
        display.db = db
        display.historic_images = []
        display.historic_offset = 0
        display.historic_mode = False
        display.image_paths = []
        display.temp_results = {}
        display.search_jsn = ""
        display.search_active = False
        display.filtered_suggestions = []
        display.selected_suggestion_idx = -1
        display.show_reset_confirm = False
        display.show_delete_confirm = False
        display.show_rebuild_confirm = False
        display.show_piece_date_dialog = False
        display.show_piece_number_dialog = False
        display.piece_number_dialog_input = ""
        display.piece_number_dialog_replace_on_input = False
        display.show_piece_identifier_dialog = False
        display.piece_identifier_dialog_input = ""
        display.piece_identifier_dialog_replace_on_input = False
        display.historic_filter_kind = ""
        display.historic_filter_label = ""
        display.historic_filter_jsns = []
        display.historic_filter_total_count = 0
        display.set_db_connection = MagicMock()
        return display

    def _build_controller(self, db=None, temp_dir=None):
        return MainController(
            display=self._build_display(db=db),
            config=ControllerConfig(temp_dir=temp_dir) if temp_dir else ControllerConfig(),
            file_manager=FileManager(),
            logger=MagicMock(),
        )

    def test_filters_independently_by_class_and_angle_with_two_digit_equality(self):
        controller = self._build_controller()
        controller.set_historic_confidence_threshold("wrinkle", "side", 0.70)
        controller.set_historic_confidence_threshold("wrinkle", "diag", 0.80)

        side_name = "100_Cam1_Side_OK.png"
        diag_name = "100_Cam2_Diag_OK.png"
        overlays = {
            side_name: [
                {"class_name": "wrinkle", "confidence": 0.704},
                {"class_name": "dent", "confidence": 0.10},
                {"class_name": "wrinkle", "confidence": None},
            ],
            diag_name: [
                {"class_name": "wrinkle", "confidence": 0.794},
                {"class_name": "wrinkle", "confidence": 0.80},
            ],
        }

        filtered, summary = controller.filter_historic_model_overlays(overlays)

        self.assertEqual(
            filtered,
            {
                side_name: overlays[side_name][:2],
                diag_name: [overlays[diag_name][1]],
            },
        )
        self.assertEqual(summary, {"visible": 3, "hidden": 2, "total": 5})

    def test_zero_threshold_removes_rule_and_reset_invalidates_render(self):
        controller = self._build_controller()
        initial_generation = controller.historic_render_generation_id

        controller.set_historic_confidence_threshold("scratch", "front", 0.456)

        self.assertEqual(
            controller.historic_confidence_thresholds,
            {("scratch", "front"): 0.46},
        )
        self.assertTrue(controller.historic_confidence_filter_active)
        self.assertGreater(controller.historic_render_generation_id, initial_generation)
        configured_generation = controller.historic_render_generation_id

        controller.set_historic_confidence_threshold("scratch", "front", 0.0)

        self.assertEqual(controller.historic_confidence_thresholds, {})
        self.assertFalse(controller.historic_confidence_filter_active)
        self.assertGreater(controller.historic_render_generation_id, configured_generation)

    def test_filter_state_survives_exit_from_historic_until_reset(self):
        controller = self._build_controller()
        controller.display.historic_mode = True
        controller.set_historic_confidence_threshold("wrinkle", "side", 0.65)

        controller.exit_historic_mode()

        self.assertTrue(controller.get_historic_confidence_filter_state()["active"])
        self.assertEqual(
            controller.get_historic_confidence_filter_state()["thresholds"],
            {("wrinkle", "side"): 0.65},
        )
        controller.reset_historic_confidence_filters()
        self.assertFalse(controller.get_historic_confidence_filter_state()["active"])

    def test_options_include_only_normalized_drawable_defects(self):
        db = MagicMock()
        db.fetch.return_value = [
            {"angle": "side", "class_name": "Wrinkle"},
            {"angle": "diag", "class_name": "dent"},
            {"angle": "side", "class_name": "wrinkle"},
            {"angle": None, "class_name": "ignored"},
        ]
        controller = self._build_controller(db=db)

        options = controller.get_historic_confidence_filter_options()

        self.assertEqual(
            options,
            [
                {"angle": "side", "class_name": "wrinkle"},
                {"angle": "diag", "class_name": "dent"},
            ],
        )
        query = db.fetch.call_args.args[0]
        self.assertIn("coordinates IS NOT NULL", query)
        self.assertIn("classification", query)

    def test_active_filter_uses_original_historic_instead_of_annotated_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            historic_dir = base / "historic"
            annotated_dir = base / "annotated"
            historic_dir.mkdir()
            annotated_dir.mkdir()
            img_name = "100_Cam1_Side_OK.png"
            historic_path = historic_dir / img_name
            annotated_path = annotated_dir / img_name
            historic_path.write_bytes(b"historic")
            annotated_path.write_bytes(b"annotated")

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "wrinkle",
                    "confidence": 0.50,
                    "model_name": "model",
                    "geometry_type": "bbox",
                    "coordinates": [1, 2, 10, 12],
                    "image_width": 20,
                    "image_height": 20,
                }
            ]
            controller = self._build_controller(db=db, temp_dir=tmp_dir)
            controller.display.historic_images = [[img_name]]
            controller._register_local_images_in_db = MagicMock()
            controller.set_historic_confidence_threshold("wrinkle", "side", 0.75)

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(result[0]["path"], str(historic_path))
            self.assertEqual(result[0]["source"], "historic_filtered")
            self.assertEqual(result[0]["status"], "ready")

    def test_active_filter_never_uses_annotated_when_original_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            (base / "historic").mkdir()
            annotated_dir = base / "annotated"
            annotated_dir.mkdir()
            img_name = "100_Cam1_Side_OK.png"
            (annotated_dir / img_name).write_bytes(b"annotated")

            controller = self._build_controller(temp_dir=tmp_dir)
            controller.display.historic_images = [[img_name]]
            controller._register_local_images_in_db = MagicMock()
            controller.set_historic_confidence_threshold("wrinkle", "side", 0.75)

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(result[0]["source"], "missing_historic_filtered")
            self.assertEqual(result[0]["status"], "missing")

    def test_visible_filtered_overlay_worker_publishes_and_reuses_render(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            historic_dir = base / "historic"
            (base / "annotated").mkdir()
            historic_dir.mkdir()
            img_name = "100_Cam1_Side_OK.png"
            cv2.imwrite(
                str(historic_dir / img_name),
                np.zeros((20, 20, 3), dtype=np.uint8),
            )

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "wrinkle",
                    "confidence": 0.90,
                    "model_name": "model",
                    "geometry_type": "bbox",
                    "coordinates": [1, 2, 10, 12],
                    "image_width": 20,
                    "image_height": 20,
                }
            ]
            controller = self._build_controller(db=db, temp_dir=tmp_dir)
            controller.display.DEFAULT_TILE_SIZE = 20
            controller.display.historic_images = [[img_name]]
            controller.display._draw_model_overlays = MagicMock(
                side_effect=lambda image, *_args: image + 1
            )
            controller._register_local_images_in_db = MagicMock()
            controller.set_historic_confidence_threshold("wrinkle", "side", 0.75)

            first = controller.download_historic_batch(tmp_dir)
            self.assertEqual(first[0]["source"], "db_coordinates+historic_filtered")
            self.assertEqual(first[0]["status"], "loading")
            controller.historic_render_worker_thread.join(timeout=2)

            second = controller.download_historic_batch(tmp_dir)

            self.assertEqual(second[0]["source"], "db_coordinates+historic_filtered")
            self.assertEqual(second[0]["status"], "ready")
            self.assertGreater(int(second[0]["prepared_image"].sum()), 0)
            controller.display._draw_model_overlays.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

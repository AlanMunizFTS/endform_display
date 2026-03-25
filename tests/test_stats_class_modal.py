import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from display_window import DisplayWindow
from main_controller import MainController


class _DisplayStatsStub:
    def __init__(self):
        self.db = MagicMock()
        self.sftp_client = None
        self.sftp_credentials = None
        self.temp_results = {}
        self._db_result_cache = {}
        self._db_registered_images = set()
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self._historic_jsn_cache = []
        self._image_cache = {}
        self.stats_class_modal_rows = []
        self.stats_class_modal_status_rows = []
        self.stats_class_modal_selected_kind = ""
        self.stats_class_modal_selected_label = ""
        self.show_stats_class_modal = False


class TestDisplayWindowStatsClassModal(unittest.TestCase):
    @patch("display_window.get_db_connection")
    def test_stats_card_single_click_opens_stats_class_modal(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.stats_card_rect = (100, 100, 200, 200)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            150,
            150,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )
        display.mouse_callback(
            cv2.EVENT_LBUTTONUP,
            150,
            150,
            0,
            None,
        )

        action_handler.assert_called_once_with("open_stats_class_modal")

    @patch("display_window.get_db_connection")
    def test_stats_card_long_press_keeps_rebuild_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.stats_card_rect = (100, 100, 200, 200)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            150,
            150,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )
        display._stats_long_press_started_at = (
            time.monotonic() - display.stats_long_press_duration_sec - 0.1
        )

        fired = display._check_stats_long_press()
        display.mouse_callback(
            cv2.EVENT_LBUTTONUP,
            150,
            150,
            0,
            None,
        )

        self.assertTrue(fired)
        action_handler.assert_called_once_with("open_rebuild_db_confirm")

    @patch("display_window.get_db_connection")
    def test_stats_class_modal_close_click_emits_close_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_close_rect = (120, 140, 100, 40)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            150,
            160,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("close_stats_class_modal")

    @patch("display_window.get_db_connection")
    def test_stats_class_modal_status_row_click_emits_detail_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_status_row_rects = [((220, 240, 120, 40), "FNOK")]

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            240,
            260,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("open_stats_status_detail", final_result="FNOK")

    @patch("display_window.get_db_connection")
    def test_stats_class_modal_blocks_background_clicks(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_close_rect = (120, 140, 100, 40)
        display.save_button_rect = (300, 300, 120, 60)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            330,
            330,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_not_called()

    @patch("display_window.get_db_connection")
    def test_draw_stats_class_modal_handles_empty_rows(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.stats_class_modal_rows = []

        canvas = np.zeros((display.height, display.width, 3), dtype=np.uint8)
        rendered = display.draw_stats_class_modal(canvas)

        self.assertIsNotNone(rendered)
        self.assertIsNotNone(display.stats_class_modal_close_rect)


class TestMainControllerStatsClassModal(unittest.TestCase):
    def test_get_piece_class_summary_queries_distinct_piece_counts(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [{"class_name": "dent", "piece_count": 3}]
        controller = MainController(display=display)

        rows = controller.get_piece_class_summary()

        self.assertEqual(rows, [{"class_name": "dent", "piece_count": 3}])
        query = display.db.fetch.call_args[0][0]
        self.assertIn("FROM piece_result_defects", query)
        self.assertIn("COUNT(DISTINCT piece_result_id) AS piece_count", query)
        self.assertIn("ORDER BY piece_count DESC, class_name ASC", query)

    def test_get_piece_status_summary_queries_final_result_counts(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [
            {"final_result": "OK", "piece_count": 8},
            {"final_result": "NOK", "piece_count": 2},
            {"final_result": "FNOK", "piece_count": 1},
            {"final_result": "FOK", "piece_count": 0},
        ]
        controller = MainController(display=display)

        rows = controller.get_piece_status_summary()

        self.assertEqual(rows[0]["final_result"], "OK")
        query = display.db.fetch.call_args[0][0]
        self.assertIn("FROM (VALUES ('OK', 1), ('NOK', 2), ('FNOK', 3), ('FOK', 4))", query)
        self.assertIn("LEFT JOIN piece_result pr ON pr.final_result = statuses.final_result", query)
        self.assertIn("ORDER BY statuses.sort_order ASC", query)

    def test_attach_historic_indices_uses_historic_piece_numbering(self):
        display = _DisplayStatsStub()
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[
                ["118610007_side_OK.png"],
                ["118610003_side_OK.png"],
                ["118610001_side_OK.png"],
            ]
        )

        rows = controller._attach_historic_indices(
            [{"jsn": "118610007"}, {"jsn": "118610001"}]
        )

        self.assertEqual(rows[0]["historic_index"], 3)
        self.assertEqual(rows[1]["historic_index"], 1)

    def test_open_stats_class_modal_populates_rows_and_shows_modal(self):
        display = _DisplayStatsStub()
        display.db.fetch.side_effect = [
            [
                {"class_name": "dent", "piece_count": 4},
                {"class_name": "scratch", "piece_count": 2},
            ],
            [
                {"final_result": "OK", "piece_count": 5},
                {"final_result": "NOK", "piece_count": 1},
                {"final_result": "FNOK", "piece_count": 0},
                {"final_result": "FOK", "piece_count": 0},
            ],
        ]
        controller = MainController(display=display)

        controller.handle_ui_action("open_stats_class_modal")

        self.assertTrue(display.show_stats_class_modal)
        self.assertEqual(
            display.stats_class_modal_rows,
            [
                {"class_name": "dent", "piece_count": 4},
                {"class_name": "scratch", "piece_count": 2},
            ],
        )
        self.assertEqual(
            display.stats_class_modal_status_rows,
            [
                {"final_result": "OK", "piece_count": 5},
                {"final_result": "NOK", "piece_count": 1},
                {"final_result": "FNOK", "piece_count": 0},
                {"final_result": "FOK", "piece_count": 0},
            ],
        )

    def test_open_stats_class_modal_handles_empty_summary(self):
        display = _DisplayStatsStub()
        display.db.fetch.side_effect = [[], []]
        controller = MainController(display=display)

        controller.handle_ui_action("open_stats_class_modal")

        self.assertTrue(display.show_stats_class_modal)
        self.assertEqual(display.stats_class_modal_rows, [])
        self.assertEqual(display.stats_class_modal_status_rows, [])

    def test_open_stats_status_detail_populates_detail_rows(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [{"jsn": "11861-0007"}, {"jsn": "11861-0003"}]
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[
                ["11861-0007_side_OK.png"],
                ["11861-0003_side_OK.png"],
            ]
        )

        controller.handle_ui_action("open_stats_status_detail", final_result="FNOK")

        self.assertEqual(display.stats_class_modal_view, "detail")
        self.assertEqual(display.stats_class_modal_selected_kind, "status")
        self.assertEqual(display.stats_class_modal_selected_label, "FNOK")
        self.assertEqual(
            display.stats_class_modal_detail_rows,
            [
                {"jsn": "11861-0007", "historic_index": 2},
                {"jsn": "11861-0003", "historic_index": 1},
            ],
        )

    def test_close_stats_class_modal_hides_modal(self):
        display = _DisplayStatsStub()
        controller = MainController(display=display)
        display.show_stats_class_modal = True

        controller.handle_ui_action("close_stats_class_modal")

        self.assertFalse(display.show_stats_class_modal)


if __name__ == "__main__":
    unittest.main(verbosity=2)

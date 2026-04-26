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
        self.stats_class_modal_matrix_rows = []
        self.stats_class_modal_view = "summary"
        self.stats_class_modal_matrix_offset = 0
        self.stats_class_modal_matrix_visible_rows = 1
        self.stats_class_modal_dataset_class_offset = 0
        self.stats_class_modal_dataset_class_visible_rows = 1
        self.stats_class_modal_dataset_result_options = []
        self.stats_class_modal_dataset_angle_options = []
        self.stats_class_modal_dataset_class_options = []
        self.stats_class_modal_dataset_selected_results = set()
        self.stats_class_modal_dataset_selected_angles = set()
        self.stats_class_modal_dataset_selected_classes = set()
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
    def test_stats_class_modal_matrix_tab_click_emits_view_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_summary_tab_rect = (120, 140, 120, 40)
        display.stats_class_modal_matrix_tab_rect = (260, 140, 120, 40)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            290,
            160,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("open_stats_matrix_view")

    @patch("display_window.get_db_connection")
    def test_stats_class_modal_matrix_export_click_emits_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_matrix_report_rect = (520, 140, 180, 36)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            560,
            160,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("export_stats_matrix_report")

    @patch("display_window.get_db_connection")
    def test_stats_class_modal_dataset_tab_click_emits_view_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_dataset_tab_rect = (400, 140, 120, 40)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            430,
            160,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("open_stats_dataset_view")

    @patch("display_window.get_db_connection")
    def test_stats_class_modal_dataset_export_click_emits_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_stats_class_modal = True
        display.stats_class_modal_dataset_export_rect = (420, 300, 220, 44)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            500,
            320,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("export_stats_dataset")

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

    @patch("display_window.get_db_connection")
    def test_draw_stats_class_modal_matrix_handles_rows(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.stats_class_modal_view = "matrix"
        display.stats_class_modal_matrix_rows = [
            {"class_name": "OK", "OK": 2, "NOK": 0, "FOK": 0, "FNOK": 0, "Total": 2},
            {"class_name": "wrinkle", "OK": 0, "NOK": 1, "FOK": 0, "FNOK": 1, "Total": 2},
            {"class_name": "Total", "OK": 2, "NOK": 1, "FOK": 0, "FNOK": 1, "Total": 4, "is_total": True},
        ]

        canvas = np.zeros((display.height, display.width, 3), dtype=np.uint8)
        rendered = display.draw_stats_class_modal(canvas)

        self.assertIsNotNone(rendered)
        self.assertIsNotNone(display.stats_class_modal_list_rect)
        self.assertIsNotNone(display.stats_class_modal_matrix_tab_rect)
        self.assertIsNotNone(display.stats_class_modal_matrix_report_rect)

    @patch("display_window.get_db_connection")
    def test_draw_stats_class_modal_dataset_handles_filter_panels(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.stats_class_modal_view = "dataset"
        display.stats_class_modal_dataset_result_options = ["OK", "NOK", "FNOK", "FOK"]
        display.stats_class_modal_dataset_angle_options = ["side", "diag", "front"]
        display.stats_class_modal_dataset_class_options = ["All", "dent", "scratch", "UNCLASSIFIED"]
        display.stats_class_modal_dataset_selected_results = {"OK", "NOK"}
        display.stats_class_modal_dataset_selected_angles = {"side", "front"}
        display.stats_class_modal_dataset_selected_classes = {"All"}

        canvas = np.zeros((display.height, display.width, 3), dtype=np.uint8)
        rendered = display.draw_stats_class_modal(canvas)

        self.assertIsNotNone(rendered)
        self.assertIsNotNone(display.stats_class_modal_dataset_tab_rect)
        self.assertIsNotNone(display.stats_class_modal_dataset_export_rect)
        self.assertTrue(display.stats_class_modal_dataset_class_rects)

    @patch("display_window.get_db_connection")
    def test_draw_stats_class_modal_detail_handles_context_columns(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.stats_class_modal_view = "detail"
        display.stats_class_modal_selected_kind = "status"
        display.stats_class_modal_selected_label = "FNOK"
        display.stats_class_modal_detail_rows = [
            {
                "jsn": "11861-0007",
                "historic_index": 2,
                "piece_date_display": "2026-03-25-09-15",
                "class_name": "wrinkle",
            }
        ]

        canvas = np.zeros((display.height, display.width, 3), dtype=np.uint8)
        rendered = display.draw_stats_class_modal(canvas)

        self.assertIsNotNone(rendered)
        self.assertIsNotNone(display.stats_class_modal_list_rect)
        self.assertIsNotNone(display.stats_class_modal_back_rect)


class TestMainControllerStatsClassModal(unittest.TestCase):
    def test_get_piece_class_summary_queries_distinct_piece_counts(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [{"class_name": "dent", "piece_count": 3}]
        controller = MainController(display=display)

        rows = controller.get_piece_class_summary()

        self.assertEqual(
            rows,
            [
                {"class_name": "dent", "piece_count": 3, "is_total": False},
                {"class_name": "Total", "piece_count": 3, "is_total": True},
            ],
        )
        query = display.db.fetch.call_args[0][0]
        self.assertIn("FROM piece_result pr", query)
        self.assertIn("LEFT JOIN piece_result_defects prd ON prd.piece_result_id = pr.id", query)
        self.assertIn("COALESCE(prd.class_name, 'UNCLASSIFIED') AS class_name", query)
        self.assertIn("COUNT(DISTINCT pr.id) AS piece_count", query)
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
        self.assertEqual(rows[-1]["final_result"], "Total")
        self.assertEqual(rows[-1]["piece_count"], 11)
        query = display.db.fetch.call_args[0][0]
        self.assertIn("FROM (VALUES ('OK', 1), ('NOK', 2), ('FNOK', 3), ('FOK', 4))", query)
        self.assertIn("LEFT JOIN piece_result pr ON pr.final_result = statuses.final_result", query)
        self.assertIn("ORDER BY statuses.sort_order ASC", query)

    def test_build_piece_stats_report_builds_matrix_and_totals(self):
        import datetime

        display = _DisplayStatsStub()
        display.db.fetch.side_effect = [
            [
                {"class_name": "wrinkle", "final_result": "NOK", "piece_count": 7},
                {"class_name": "OK", "final_result": "OK", "piece_count": 15},
                {"class_name": "wrinkle", "final_result": "FNOK", "piece_count": 5},
                {"class_name": "split", "final_result": "NOK", "piece_count": 5},
                {"class_name": "split", "final_result": "FNOK", "piece_count": 3},
                {"class_name": "UNCLASSIFIED", "final_result": "FOK", "piece_count": 2},
            ],
            [
                {
                    "start_at": datetime.datetime(2026, 3, 25, 9, 0),
                    "end_at": datetime.datetime(2026, 3, 25, 12, 0),
                }
            ],
        ]
        controller = MainController(display=display)

        report = controller.build_piece_stats_report()

        self.assertEqual(report["rows"][0]["class_name"], "OK")
        self.assertEqual(report["rows"][0]["Total"], 15)
        self.assertEqual(report["rows"][1]["class_name"], "wrinkle")
        self.assertEqual(report["rows"][1]["FNOK"], 5)
        self.assertEqual(report["rows"][-2]["class_name"], "UNCLASSIFIED")
        self.assertEqual(report["rows"][-1]["class_name"], "Total")
        self.assertEqual(report["rows"][-1]["OK"], 15)
        self.assertEqual(report["rows"][-1]["NOK"], 12)
        self.assertEqual(report["rows"][-1]["FOK"], 2)
        self.assertEqual(report["rows"][-1]["FNOK"], 8)
        self.assertEqual(report["rows"][-1]["Total"], 37)

    def test_build_piece_stats_report_includes_unclassified_when_no_defect_row_exists(self):
        import datetime

        display = _DisplayStatsStub()
        display.db.fetch.side_effect = [
            [
                {"class_name": "UNCLASSIFIED", "final_result": "OK", "piece_count": 4},
            ],
            [
                {
                    "start_at": datetime.datetime(2026, 3, 25, 9, 0),
                    "end_at": datetime.datetime(2026, 3, 25, 10, 0),
                }
            ],
        ]
        controller = MainController(display=display)

        report = controller.build_piece_stats_report()

        self.assertEqual(report["rows"][0]["class_name"], "UNCLASSIFIED")
        self.assertEqual(report["rows"][0]["OK"], 4)
        self.assertEqual(report["rows"][-1]["Total"], 4)

    def test_get_piece_stats_dataset_filter_options_includes_unclassified_and_all(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [
            {"class_name": "dent"},
            {"class_name": "scratch"},
        ]
        controller = MainController(display=display)

        options = controller.get_piece_stats_dataset_filter_options()

        self.assertEqual(options["results"], ["OK", "NOK", "FOK", "FNOK"])
        self.assertEqual(options["angles"], ["side", "diag", "front"])
        self.assertEqual(options["class_names"][0], "All")
        self.assertIn("UNCLASSIFIED", options["class_names"])

    def test_get_piece_stats_dataset_records_builds_image_level_rows(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [
            {
                "img_name": "11861_Cam1_Side1_OK.png",
                "operator_result": "OK",
                "model_result": "OK",
                "class_name": None,
            },
            {
                "img_name": "11861_Cam2_Front_NOK.png",
                "operator_result": "NOK",
                "model_result": "OK",
                "class_name": "dent",
            },
            {
                "img_name": "11861_Cam2_Front_NOK.png",
                "operator_result": "NOK",
                "model_result": "OK",
                "class_name": "scratch",
            },
        ]
        controller = MainController(display=display)

        rows = controller.get_piece_stats_dataset_records()

        self.assertEqual(
            rows,
            [
                {
                    "img_name": "11861_Cam1_Side1_OK.png",
                    "result": "OK",
                    "angle": "side",
                    "class_names": ["UNCLASSIFIED"],
                },
                {
                    "img_name": "11861_Cam2_Front_NOK.png",
                    "result": "FOK",
                    "angle": "front",
                    "class_names": ["dent", "scratch"],
                },
            ],
        )

    def test_toggle_piece_stats_dataset_class_keeps_all_semantics(self):
        display = _DisplayStatsStub()
        controller = MainController(display=display)
        display.stats_class_modal_dataset_class_options = ["All", "dent", "scratch", "UNCLASSIFIED"]
        display.stats_class_modal_dataset_selected_classes = {"All"}

        controller.handle_ui_action("toggle_stats_dataset_class", value="dent")
        self.assertEqual(display.stats_class_modal_dataset_selected_classes, {"dent"})

        controller.handle_ui_action("toggle_stats_dataset_class", value="scratch")
        self.assertEqual(display.stats_class_modal_dataset_selected_classes, {"dent", "scratch"})

        controller.handle_ui_action("toggle_stats_dataset_class", value="dent")
        controller.handle_ui_action("toggle_stats_dataset_class", value="scratch")
        self.assertEqual(display.stats_class_modal_dataset_selected_classes, {"All"})

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

    def test_get_piece_jsns_for_class_includes_date_and_final_result(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [
            {
                "jsn": "118610007",
                "created_at": "2026-03-25 09:15:33",
                "final_result": "FNOK",
            }
        ]
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[["118610007_side_OK.png"]]
        )

        rows = controller.get_piece_jsns_for_class("wrinkle")

        self.assertEqual(
            rows,
            [
                {
                    "jsn": "118610007",
                    "created_at": "2026-03-25 09:15:33",
                    "final_result": "FNOK",
                    "historic_index": 1,
                    "piece_date_display": "2026-03-25-09-15",
                }
            ],
        )
        query = display.db.fetch.call_args[0][0]
        self.assertIn("COALESCE(prd.class_name, 'UNCLASSIFIED') = %s", query)
        self.assertIn("pr.created_at", query)
        self.assertIn("pr.final_result", query)

    def test_get_piece_jsns_for_status_includes_date_and_defect(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [
            {
                "jsn": "118610007",
                "created_at": "2026-03-25 09:15:33",
                "class_name": "UNCLASSIFIED",
            }
        ]
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[["118610007_side_OK.png"]]
        )

        rows = controller.get_piece_jsns_for_status("FNOK")

        self.assertEqual(
            rows,
            [
                {
                    "jsn": "118610007",
                    "created_at": "2026-03-25 09:15:33",
                    "class_name": "UNCLASSIFIED",
                    "historic_index": 1,
                    "piece_date_display": "2026-03-25-09-15",
                }
            ],
        )
        query = display.db.fetch.call_args[0][0]
        self.assertIn("COALESCE(prd.class_name, 'UNCLASSIFIED') AS class_name", query)
        self.assertIn("pr.created_at", query)

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
            [
                {"class_name": "OK", "final_result": "OK", "piece_count": 5},
                {"class_name": "scratch", "final_result": "NOK", "piece_count": 1},
            ],
            [{"start_at": None, "end_at": None}],
            [{"class_name": "scratch"}],
        ]
        controller = MainController(display=display)

        controller.handle_ui_action("open_stats_class_modal")

        self.assertTrue(display.show_stats_class_modal)
        self.assertEqual(
            display.stats_class_modal_rows,
            [
                {"class_name": "dent", "piece_count": 4, "is_total": False},
                {"class_name": "scratch", "piece_count": 2, "is_total": False},
                {"class_name": "Total", "piece_count": 6, "is_total": True},
            ],
        )
        self.assertEqual(
            display.stats_class_modal_status_rows,
            [
                {"final_result": "OK", "piece_count": 5, "is_total": False},
                {"final_result": "NOK", "piece_count": 1, "is_total": False},
                {"final_result": "FNOK", "piece_count": 0, "is_total": False},
                {"final_result": "FOK", "piece_count": 0, "is_total": False},
                {"final_result": "Total", "piece_count": 6, "is_total": True},
            ],
        )
        self.assertEqual(display.stats_class_modal_matrix_rows[-1]["class_name"], "Total")
        self.assertEqual(display.stats_class_modal_dataset_selected_classes, {"All"})
        self.assertIn("UNCLASSIFIED", display.stats_class_modal_dataset_class_options)

    def test_open_stats_class_modal_handles_empty_summary(self):
        display = _DisplayStatsStub()
        display.db.fetch.side_effect = [[], [], [], [{"start_at": None, "end_at": None}], []]
        controller = MainController(display=display)

        controller.handle_ui_action("open_stats_class_modal")

        self.assertTrue(display.show_stats_class_modal)
        self.assertEqual(display.stats_class_modal_rows, [])
        self.assertEqual(display.stats_class_modal_status_rows, [])
        self.assertEqual(display.stats_class_modal_matrix_rows, [])

    def test_open_stats_status_detail_populates_detail_rows(self):
        display = _DisplayStatsStub()
        display.db.fetch.return_value = [
            {
                "jsn": "118610007",
                "created_at": "2026-03-25 09:15:33",
                "class_name": "wrinkle",
            },
            {
                "jsn": "118610003",
                "created_at": "2026-03-25 08:45:00",
                "class_name": "split",
            },
        ]
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[
                ["118610007_side_OK.png"],
                ["118610003_side_OK.png"],
            ]
        )

        controller.handle_ui_action("open_stats_status_detail", final_result="FNOK")

        self.assertEqual(display.stats_class_modal_view, "detail")
        self.assertEqual(display.stats_class_modal_selected_kind, "status")
        self.assertEqual(display.stats_class_modal_selected_label, "FNOK")
        self.assertEqual(
            display.stats_class_modal_detail_rows,
            [
                {
                    "jsn": "118610007",
                    "created_at": "2026-03-25 09:15:33",
                    "class_name": "wrinkle",
                    "historic_index": 2,
                    "piece_date_display": "2026-03-25-09-15",
                },
                {
                    "jsn": "118610003",
                    "created_at": "2026-03-25 08:45:00",
                    "class_name": "split",
                    "historic_index": 1,
                    "piece_date_display": "2026-03-25-08-45",
                },
            ],
        )

    def test_close_stats_class_modal_hides_modal(self):
        display = _DisplayStatsStub()
        controller = MainController(display=display)
        display.show_stats_class_modal = True

        controller.handle_ui_action("close_stats_class_modal")

        self.assertFalse(display.show_stats_class_modal)

    def test_open_stats_matrix_view_switches_modal_view(self):
        display = _DisplayStatsStub()
        controller = MainController(display=display)

        controller.handle_ui_action("open_stats_matrix_view")

        self.assertEqual(display.stats_class_modal_view, "matrix")

    def test_open_stats_dataset_view_switches_modal_view(self):
        display = _DisplayStatsStub()
        controller = MainController(display=display)

        controller.handle_ui_action("open_stats_dataset_view")

        self.assertEqual(display.stats_class_modal_view, "dataset")


if __name__ == "__main__":
    unittest.main(verbosity=2)

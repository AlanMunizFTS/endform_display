import unittest
from unittest.mock import MagicMock, patch

import cv2

from display_window import DisplayWindow


class TestDisplayWindowStatePackage(unittest.TestCase):
    @patch("display_window.get_db_connection")
    def test_export_button_click_emits_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.export_button_rect = (200, 200, 180, 60)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            220,
            220,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("export_display_state")

    @patch("display_window.get_db_connection")
    def test_image_report_button_click_opens_nonblocking_dialog(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.historic_mode = True
        display.image_report_button_rect = (200, 200, 180, 60)

        with patch.object(
            display,
            "_open_historic_image_report_dialog",
            return_value=True,
        ) as open_dialog:
            display.mouse_callback(
                cv2.EVENT_LBUTTONDOWN,
                220,
                220,
                cv2.EVENT_FLAG_LBUTTON,
                None,
            )

        open_dialog.assert_called_once_with()
        action_handler.assert_not_called()

    @patch("display_window.get_db_connection")
    def test_image_report_dialog_result_dispatches_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display._image_report_dialog_result = {
            "endform_type": "mush",
            "class_name": "wrinkle",
            "defect_class": "wrinkle",
            "angle": "diag",
        }

        display._pump_historic_image_report_dialog()

        action_handler.assert_called_once_with(
            "export_historic_image_report",
            endform_type="mush",
            class_name="wrinkle",
            defect_class="wrinkle",
            angle="diag",
        )

    @patch("display_window.get_db_connection")
    def test_image_report_dialog_analysis_result_dispatches_action(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display._image_report_dialog_result = {
            "_action": "open_historic_verdict_analysis",
            "endform_type": "mush",
            "class_name": "wrinkle",
            "defect_class": "wrinkle",
            "angle": "side",
        }

        display._pump_historic_image_report_dialog()

        action_handler.assert_called_once_with(
            "open_historic_verdict_analysis",
            endform_type="mush",
            class_name="wrinkle",
            defect_class="wrinkle",
            angle="side",
        )

    @patch("display_window.get_db_connection")
    def test_verdict_analysis_paste_is_atomic_and_starts_at_selected_row(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display._verdict_analysis_rows = [
            {"actual_result": "", "positions": []},
            {"actual_result": "", "positions": []},
            {"actual_result": "", "positions": []},
        ]

        applied = display._apply_historic_verdict_paste("ok\nNOK", start_index=1)

        self.assertEqual(applied, 2)
        self.assertEqual(
            [row["actual_result"] for row in display._verdict_analysis_rows],
            ["", "OK", "NOK"],
        )
        before_invalid_paste = list(
            row["actual_result"] for row in display._verdict_analysis_rows
        )
        with self.assertRaises(ValueError):
            display._apply_historic_verdict_paste("OK\nINVALID", start_index=0)
        self.assertEqual(
            [row["actual_result"] for row in display._verdict_analysis_rows],
            before_invalid_paste,
        )
        with self.assertRaisesRegex(ValueError, "only 1 rows"):
            display._apply_historic_verdict_paste("OK\nNOK", start_index=2)
        self.assertEqual(
            [row["actual_result"] for row in display._verdict_analysis_rows],
            before_invalid_paste,
        )

    @patch("display_window.get_db_connection")
    def test_closing_verdict_analysis_discards_session_values(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display._verdict_analysis_rows = [{"actual_result": "OK", "positions": []}]
        display._verdict_analysis_dirty = True

        closed = display._close_historic_verdict_analysis_dialog(confirm=False)

        self.assertTrue(closed)
        self.assertEqual(display._verdict_analysis_rows, [])
        self.assertFalse(display._verdict_analysis_dirty)

    @patch("display_window.get_db_connection")
    def test_image_report_dialog_without_result_does_not_emit_action(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)

        display._pump_historic_image_report_dialog()

        action_handler.assert_not_called()

    @patch("display_window.get_db_connection")
    def test_image_report_filter_options_come_from_model_results(
        self,
        mock_get_db_connection,
    ):
        db = MagicMock()
        db.fetch.return_value = [
            {"angle": "diag", "class_name": "wrinkle"},
            {"angle": "side", "class_name": "wrinkle"},
            {"angle": "side", "class_name": "dent"},
            {"angle": "side", "class_name": "dent"},
        ]
        mock_get_db_connection.return_value = db
        display = DisplayWindow(file_manager=MagicMock())

        options = display._get_historic_image_report_filter_options()

        self.assertEqual(
            options,
            [
                {"angle": "side", "class_name": "dent"},
                {"angle": "side", "class_name": "wrinkle"},
                {"angle": "diag", "class_name": "wrinkle"},
            ],
        )
        query = db.fetch.call_args[0][0]
        self.assertIn("FROM model_results", query)
        self.assertIn("coordinates IS NOT NULL", query)

    @patch("display_window.get_db_connection")
    def test_import_button_click_emits_action_with_selected_path(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.import_button_rect = (200, 200, 180, 60)

        with patch.object(
            display,
            "_choose_import_package_path",
            return_value="C:\\tmp\\display_state_20260326_120000",
        ):
            display.mouse_callback(
                cv2.EVENT_LBUTTONDOWN,
                220,
                220,
                cv2.EVENT_FLAG_LBUTTON,
                None,
            )

        action_handler.assert_called_once_with(
            "import_display_state",
            package_path="C:\\tmp\\display_state_20260326_120000",
        )

    @patch("display_window.get_db_connection")
    def test_import_button_cancel_does_not_emit_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.import_button_rect = (200, 200, 180, 60)

        with patch.object(display, "_choose_import_package_path", return_value=None):
            display.mouse_callback(
                cv2.EVENT_LBUTTONDOWN,
                220,
                220,
                cv2.EVENT_FLAG_LBUTTON,
                None,
            )

        action_handler.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

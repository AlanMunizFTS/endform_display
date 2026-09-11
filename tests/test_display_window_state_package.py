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
    def test_confidence_button_click_opens_nonblocking_dialog(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.historic_mode = True
        display.confidence_button_rect = (200, 200, 180, 60)

        with patch.object(
            display,
            "_open_historic_confidence_dialog",
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
    def test_closing_confidence_dialog_keeps_controller_filters(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        controller = MagicMock()
        root = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), controller=controller)
        display._historic_confidence_dialog_root = root

        display._close_historic_confidence_dialog()

        root.destroy.assert_called_once_with()
        controller.reset_historic_confidence_filters.assert_not_called()
        self.assertIsNone(display._historic_confidence_dialog_root)

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
            "angle": "side+diag",
        }

        display._pump_historic_image_report_dialog()

        action_handler.assert_called_once_with(
            "open_historic_verdict_analysis",
            endform_type="mush",
            class_name="wrinkle",
            defect_class="wrinkle",
            angle="side+diag",
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
    def test_verdict_analysis_copies_one_inferred_position_as_excel_column(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display._verdict_analysis_rows = [
            {
                "positions": [
                    {"inferred_result": "OK"},
                    {"inferred_result": "nok"},
                    {"inferred_result": None},
                ]
            },
            {"positions": [{"inferred_result": "NOK"}]},
        ]

        expected = "NOK\r\nN/D"
        self.assertEqual(
            display._build_historic_inferred_column_tsv(2),
            expected,
        )
        with patch.object(display, "_copy_text_to_clipboard", return_value=True) as copy_mock:
            self.assertTrue(display._copy_historic_inferred_column(2))

        copy_mock.assert_called_once_with(expected)

    @patch("display_window.get_db_connection")
    def test_verdict_analysis_exports_with_current_thresholds(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(
            file_manager=MagicMock(),
            action_handler=action_handler,
        )
        display._verdict_analysis_confidence_thresholds = {
            "side": 0.56,
            "diag": 0.43,
        }

        payload = display._export_historic_verdict_analysis_report(
            {
                "endform_type": "mush",
                "defect_class": "edge",
                "angle": "side+diag",
                "pieces_per_group": 4,
            }
        )

        self.assertEqual(
            payload["confidence_thresholds"],
            {"side": 0.56, "diag": 0.43},
        )
        action_handler.assert_called_once_with(
            "export_historic_image_report",
            endform_type="mush",
            class_name="edge",
            defect_class="edge",
            angle="side+diag",
            pieces_per_group=4,
            confidence_thresholds={"side": 0.56, "diag": 0.43},
        )

    @patch("display_window.get_db_connection")
    def test_closing_verdict_analysis_discards_session_values(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display._verdict_analysis_rows = [{"actual_result": "OK", "positions": []}]
        display._verdict_analysis_required_angles = ("side", "diag")
        display._verdict_analysis_confidence_thresholds = {
            "side": 0.45,
            "diag": 0.60,
        }
        display._verdict_analysis_dirty = True

        closed = display._close_historic_verdict_analysis_dialog(confirm=False)

        self.assertTrue(closed)
        self.assertEqual(display._verdict_analysis_rows, [])
        self.assertEqual(display._verdict_analysis_required_angles, ())
        self.assertEqual(display._verdict_analysis_confidence_thresholds, {})
        self.assertIsNone(display._verdict_analysis_overall_accuracy_var)
        self.assertFalse(display._verdict_analysis_dirty)

    @patch("display_window.get_db_connection")
    def test_verdict_analysis_threshold_change_recalculates_without_changing_actual(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display._verdict_analysis_required_angles = ("side", "diag")
        display._verdict_analysis_confidence_thresholds = {
            "side": 0.0,
            "diag": 0.0,
        }
        display._verdict_analysis_rows = [
            {
                "actual_result": "NOK",
                "positions": [
                    {
                        "position": 1,
                        "jsn": "jsn-1",
                        "inferred_result": "NOK",
                        "confidence_data_complete": True,
                        "max_confidence_by_angle": {
                            "side": 0.60,
                            "diag": 0.20,
                        },
                    }
                ],
            }
        ]

        display._set_historic_verdict_confidence_thresholds(
            {"side": 0.70, "diag": 0.30}
        )

        self.assertEqual(display._verdict_analysis_rows[0]["actual_result"], "NOK")
        self.assertEqual(
            display._verdict_analysis_rows[0]["positions"][0]["inferred_result"],
            "OK",
        )
        self.assertFalse(display._verdict_analysis_dirty)

    @patch("display_window.get_db_connection")
    def test_verdict_analysis_refreshes_position_and_overall_accuracy(
        self,
        mock_get_db_connection,
    ):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display._verdict_analysis_rows = [
            {
                "actual_result": "OK",
                "positions": [
                    {"position": 1, "jsn": "ok-1", "inferred_result": "OK"},
                    {"position": 2, "jsn": "nok-1", "inferred_result": "NOK"},
                ],
            },
            {
                "actual_result": "NOK",
                "positions": [
                    {"position": 1, "jsn": "nok-2", "inferred_result": "NOK"},
                    {"position": 2, "jsn": "nok-3", "inferred_result": "NOK"},
                ],
            },
        ]
        position_1_accuracy = MagicMock()
        position_2_accuracy = MagicMock()
        overall_accuracy = MagicMock()
        display._verdict_analysis_metric_vars = {
            (1, "accuracy"): position_1_accuracy,
            (2, "accuracy"): position_2_accuracy,
        }
        display._verdict_analysis_overall_accuracy_var = overall_accuracy

        display._refresh_historic_verdict_analysis()

        position_1_accuracy.set.assert_called_once_with("100.00% (2/2)")
        position_2_accuracy.set.assert_called_once_with("50.00% (1/2)")
        overall_accuracy.set.assert_called_once_with(
            "Overall Accuracy (average of positions): 75.00%"
        )

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
            {"angle": "side", "class_name": "Nylon_In_Form"},
        ]
        mock_get_db_connection.return_value = db
        display = DisplayWindow(file_manager=MagicMock())

        options = display._get_historic_image_report_filter_options()

        self.assertEqual(
            options,
            [
                {"angle": "side", "class_name": "dent"},
                {"angle": "side", "class_name": "nylon_in_form"},
                {"angle": "side", "class_name": "wrinkle"},
                {"angle": "diag", "class_name": "wrinkle"},
                {"angle": "side+diag", "class_name": "wrinkle"},
            ],
        )
        query = db.fetch.call_args[0][0]
        self.assertIn("FROM model_results", query)
        self.assertIn("coordinates IS NOT NULL", query)
        self.assertIn("geometry_type", query)
        self.assertIn("classification", query)

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

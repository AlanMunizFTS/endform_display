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
    def test_import_button_click_emits_action_with_selected_path(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.import_button_rect = (200, 200, 180, 60)

        with patch.object(
            display,
            "_choose_import_package_path",
            return_value="C:\\tmp\\display_state.zip",
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
            package_path="C:\\tmp\\display_state.zip",
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

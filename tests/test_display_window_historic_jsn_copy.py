import unittest
from unittest.mock import MagicMock, patch

import cv2

from display_window import DisplayWindow


class TestDisplayWindowHistoricJsnCopy(unittest.TestCase):
    @patch("display_window.get_db_connection")
    def test_copy_current_historic_jsn_sets_success_toast(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.historic_mode = True
        display.historic_images = [["11861_side_cam1.png"]]
        display.historic_offset = 0

        with patch.object(display, "_copy_text_to_clipboard", return_value=True) as copy_mock:
            copied = display._copy_current_historic_jsn()

        self.assertTrue(copied)
        copy_mock.assert_called_once_with("11861")
        self.assertEqual(display.toast_message, "Copied JSN 11861")
        self.assertFalse(display.toast_message_is_error)

    @patch("display_window.get_db_connection")
    def test_copy_current_historic_jsn_without_batch_sets_error_toast(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.historic_mode = True
        display.historic_images = []
        display.historic_offset = 0

        copied = display._copy_current_historic_jsn()

        self.assertFalse(copied)
        self.assertEqual(display.toast_message, "No JSN available to copy")
        self.assertTrue(display.toast_message_is_error)

    @patch("display_window.get_db_connection")
    def test_mouse_click_on_historic_jsn_banner_triggers_copy(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        display = DisplayWindow(file_manager=MagicMock())
        display.historic_mode = True
        display.historic_jsn_rect = (100, 20, 200, 50)

        with patch.object(display, "_copy_current_historic_jsn") as copy_mock:
            display.mouse_callback(
                cv2.EVENT_LBUTTONDOWN,
                150,
                40,
                cv2.EVENT_FLAG_LBUTTON,
                None,
            )

        copy_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)

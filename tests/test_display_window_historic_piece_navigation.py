import unittest
from unittest.mock import MagicMock, patch

import cv2

from display_window import DisplayWindow


class TestDisplayWindowHistoricPieceNavigation(unittest.TestCase):
    @patch("display_window.get_db_connection")
    def test_mouse_click_on_piece_counter_opens_dialog(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.historic_mode = True
        display.piece_counter_rect = (100, 100, 220, 48)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            150,
            120,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("open_piece_number_dialog")

    @patch("display_window.get_db_connection")
    def test_piece_number_dialog_keyboard_digit_emits_append_action(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.image = MagicMock()
        display.show_piece_number_dialog = True

        with patch("display_window.cv2.namedWindow"), \
             patch("display_window.cv2.setWindowProperty"), \
             patch("display_window.cv2.setMouseCallback"), \
             patch("display_window.cv2.imshow"), \
             patch("display_window.cv2.waitKeyEx", return_value=ord("5")):
            refreshed = display.show()

        self.assertTrue(refreshed)
        action_handler.assert_called_once_with("piece_number_append_digit", digit="5")

    @patch("display_window.get_db_connection")
    def test_piece_number_dialog_ok_button_submits(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.show_piece_number_dialog = True
        display.piece_number_dialog_ok_rect = (200, 200, 120, 42)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            240,
            220,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("submit_piece_number_dialog")

    @patch("display_window.get_db_connection")
    def test_mouse_click_on_piece_identifier_opens_dialog(self, mock_get_db_connection):
        mock_get_db_connection.return_value = MagicMock()
        action_handler = MagicMock()
        display = DisplayWindow(file_manager=MagicMock(), action_handler=action_handler)
        display.historic_mode = True
        display.piece_identifier_rect = (100, 100, 120, 42)

        display.mouse_callback(
            cv2.EVENT_LBUTTONDOWN,
            150,
            120,
            cv2.EVENT_FLAG_LBUTTON,
            None,
        )

        action_handler.assert_called_once_with("open_piece_identifier_dialog")


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from display_window import DisplayWindow


class TestDisplayWindowSFTPState(unittest.TestCase):
    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_constructor_respects_initial_sftp_client_without_remote_controls(self, _db_mock):
        fake_client = object()
        window = DisplayWindow(sftp_client=fake_client)

        self.assertIs(window.sftp_client, fake_client)
        self.assertFalse(hasattr(window, "remote_controls_enabled"))
        self.assertFalse(hasattr(window, "remote_requested"))
        self.assertFalse(hasattr(window, "trigger_active"))

    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_set_sftp_client_only_updates_sftp_client(self, _db_mock):
        window = DisplayWindow(sftp_client=None)
        self.assertIsNone(window.sftp_client)

        fake_client = object()
        window.set_sftp_client(fake_client)
        self.assertIs(window.sftp_client, fake_client)

        window.set_sftp_client(None)

        self.assertIsNone(window.sftp_client)
        self.assertFalse(hasattr(window, "capture_modal_visible"))

    @patch("display_window.cv2.imshow")
    @patch("display_window.cv2.setMouseCallback")
    @patch("display_window.cv2.setWindowProperty")
    @patch("display_window.cv2.namedWindow")
    @patch("display_window.cv2.waitKeyEx", return_value=13)
    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_enter_key_in_normal_mode_does_not_emit_remote_input(
        self,
        _db_mock,
        _wait_key,
        _named_window,
        _set_window_property,
        _set_mouse_callback,
        _imshow,
    ):
        action_handler = MagicMock()
        window = DisplayWindow(refresh_interval=0, action_handler=action_handler)
        window.image = np.zeros((window.height, window.width, 3), dtype=np.uint8)

        self.assertTrue(window.show())

        action_handler.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

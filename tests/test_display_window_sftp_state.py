import unittest
from unittest.mock import MagicMock, patch

from display_window import DisplayWindow


class TestDisplayWindowSFTPState(unittest.TestCase):
    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_constructor_respects_initial_sftp_client(self, _db_mock):
        fake_client = object()
        window = DisplayWindow(sftp_client=fake_client)

        self.assertIs(window.sftp_client, fake_client)
        self.assertTrue(window.remote_controls_enabled)

    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_set_sftp_client_enables_and_disables_controls(self, _db_mock):
        window = DisplayWindow(sftp_client=None)
        self.assertFalse(window.remote_controls_enabled)

        fake_client = object()
        window.set_sftp_client(fake_client)
        self.assertIs(window.sftp_client, fake_client)
        self.assertTrue(window.remote_controls_enabled)

        window.remote_requested = True
        window.remote_action_request = "start"
        window.trigger_active = True
        window.connected_cameras = {"cam_1"}
        window.set_sftp_client(None)

        self.assertIsNone(window.sftp_client)
        self.assertFalse(window.remote_controls_enabled)
        self.assertFalse(window.remote_requested)
        self.assertIsNone(window.remote_action_request)
        self.assertFalse(window.trigger_active)
        self.assertEqual(window.connected_cameras, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)

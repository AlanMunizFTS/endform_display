import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from display_window import DisplayWindow
from paths_config import HISTORIC_SUBDIR_NAME


class TestDisplayWindowHistoricDownload(unittest.TestCase):
    @patch("display_window.get_db_connection", return_value=MagicMock())
    @patch("display_window.Event")
    @patch("display_window.Process")
    def test_start_historic_download_without_live_sftp_client(
        self, process_cls, event_cls, _db_mock
    ):
        fake_event = MagicMock()
        event_cls.return_value = fake_event
        fake_process = MagicMock()
        process_cls.return_value = fake_process

        creds = {
            "hostname": "host",
            "port": 22,
            "username": "user",
            "password": "pwd",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            window = DisplayWindow(sftp_client=None, sftp_credentials=creds)
            window.start_historic_download_on_startup(tmpdir, check_interval=11)

            self.assertTrue(os.path.isdir(os.path.join(tmpdir, HISTORIC_SUBDIR_NAME)))
            event_cls.assert_called_once()
            process_cls.assert_called_once()
            args = process_cls.call_args.kwargs["args"]
            self.assertEqual(args[0], "host")
            self.assertEqual(args[1], 22)
            self.assertEqual(args[2], "user")
            self.assertEqual(args[3], "pwd")
            self.assertEqual(args[6], 11)
            self.assertEqual(args[7], 10)
            self.assertIs(args[8], fake_event)
            self.assertTrue(fake_process.daemon)
            fake_process.start.assert_called_once()

    @patch("display_window.get_db_connection", return_value=MagicMock())
    @patch("display_window.Event")
    @patch("display_window.Process")
    def test_start_historic_download_skips_when_credentials_missing(
        self, process_cls, event_cls, _db_mock
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            window = DisplayWindow(sftp_client=None, sftp_credentials=None)
            window.start_historic_download_on_startup(tmpdir, check_interval=10)

            self.assertTrue(os.path.isdir(os.path.join(tmpdir, HISTORIC_SUBDIR_NAME)))
            event_cls.assert_not_called()
            process_cls.assert_not_called()

    @patch("display_window.get_db_connection", return_value=MagicMock())
    @patch("display_window.Event")
    @patch("display_window.Process")
    def test_start_historic_download_skips_when_worker_already_alive(
        self, process_cls, event_cls, _db_mock
    ):
        creds = {
            "hostname": "host",
            "port": 22,
            "username": "user",
            "password": "pwd",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            window = DisplayWindow(sftp_client=None, sftp_credentials=creds)
            existing_process = MagicMock()
            existing_process.is_alive.return_value = True
            window.download_process = existing_process
            window.start_historic_download_on_startup(tmpdir, check_interval=10)

            event_cls.assert_not_called()
            process_cls.assert_not_called()

    @patch("display_window.get_db_connection", return_value=MagicMock())
    @patch("display_window.cv2.destroyWindow")
    def test_close_stops_background_worker(self, _destroy_window, _db_mock):
        window = DisplayWindow(sftp_client=None, sftp_credentials=None)
        fake_event = MagicMock()
        fake_process = MagicMock()
        fake_process.is_alive.return_value = True
        window.download_stop_event = fake_event
        window.download_process = fake_process

        window.close()

        fake_event.set.assert_called_once()
        fake_process.join.assert_called()
        fake_process.terminate.assert_called_once()
        self.assertIsNone(window.download_process)
        self.assertIsNone(window.download_stop_event)


if __name__ == "__main__":
    unittest.main(verbosity=2)

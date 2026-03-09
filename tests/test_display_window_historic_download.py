import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from display_window import DisplayWindow
from paths_config import ANNOTATED_SUBDIR_NAME, HISTORIC_SUBDIR_NAME


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
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, ANNOTATED_SUBDIR_NAME)))
            self.assertEqual(event_cls.call_count, 2)
            self.assertEqual(process_cls.call_count, 2)

            historic_args = process_cls.call_args_list[0].kwargs["args"]
            self.assertEqual(historic_args[0], "host")
            self.assertEqual(historic_args[1], 22)
            self.assertEqual(historic_args[2], "user")
            self.assertEqual(historic_args[3], "pwd")
            self.assertEqual(historic_args[6], 11)
            self.assertEqual(historic_args[7], 10)
            self.assertIs(historic_args[8], fake_event)
            self.assertEqual(historic_args[9], "HIST_SYNC_SSH")

            annotated_args = process_cls.call_args_list[1].kwargs["args"]
            self.assertEqual(annotated_args[0], "host")
            self.assertEqual(annotated_args[1], 22)
            self.assertEqual(annotated_args[2], "user")
            self.assertEqual(annotated_args[3], "pwd")
            self.assertEqual(annotated_args[6], 11)
            self.assertEqual(annotated_args[7], 10)
            self.assertEqual(annotated_args[9], "ANNOTATED_SYNC_SSH")

            self.assertTrue(fake_process.daemon)
            self.assertEqual(fake_process.start.call_count, 2)

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
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, ANNOTATED_SUBDIR_NAME)))
            event_cls.assert_not_called()
            process_cls.assert_not_called()

    @patch("display_window.get_db_connection", return_value=MagicMock())
    @patch("display_window.Event")
    @patch("display_window.Process")
    def test_start_historic_download_skips_when_all_workers_already_alive(
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
            existing_historic_process = MagicMock()
            existing_historic_process.is_alive.return_value = True
            existing_annotated_process = MagicMock()
            existing_annotated_process.is_alive.return_value = True
            window.download_process = existing_historic_process
            window.annotated_download_process = existing_annotated_process
            window.start_historic_download_on_startup(tmpdir, check_interval=10)

            event_cls.assert_not_called()
            process_cls.assert_not_called()

    @patch("display_window.get_db_connection", return_value=MagicMock())
    @patch("display_window.cv2.destroyWindow")
    def test_close_stops_background_worker(self, _destroy_window, _db_mock):
        window = DisplayWindow(sftp_client=None, sftp_credentials=None)
        fake_historic_event = MagicMock()
        fake_historic_process = MagicMock()
        fake_historic_process.is_alive.return_value = True
        fake_annotated_event = MagicMock()
        fake_annotated_process = MagicMock()
        fake_annotated_process.is_alive.return_value = True
        window.download_stop_event = fake_historic_event
        window.download_process = fake_historic_process
        window.annotated_download_stop_event = fake_annotated_event
        window.annotated_download_process = fake_annotated_process

        window.close()

        fake_historic_event.set.assert_called_once()
        fake_historic_process.join.assert_called()
        fake_historic_process.terminate.assert_called_once()
        fake_annotated_event.set.assert_called_once()
        fake_annotated_process.join.assert_called()
        fake_annotated_process.terminate.assert_called_once()
        self.assertIsNone(window.download_process)
        self.assertIsNone(window.download_stop_event)
        self.assertIsNone(window.annotated_download_process)
        self.assertIsNone(window.annotated_download_stop_event)


if __name__ == "__main__":
    unittest.main(verbosity=2)

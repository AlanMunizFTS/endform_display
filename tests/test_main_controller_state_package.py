import tempfile
import unittest
from unittest.mock import MagicMock, patch

from main_controller import ControllerConfig, MainController


class _ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class _AliveThread:
    def is_alive(self):
        return True


class TestMainControllerStatePackage(unittest.TestCase):
    def _build_display(self, db=None):
        display = MagicMock()
        display.db = db
        display.sftp_client = None
        display.sftp_credentials = None
        display.sync_in_progress = False
        display.reset_in_progress = False
        display.sync_progress = 0
        display.sync_stage = ""
        display.sync_progress_title = ""
        display.sync_progress_helper_text = ""
        display.sync_message = ""
        display.sync_message_is_error = False
        display.sync_message_time = 0
        display.set_db_connection = MagicMock()
        return display

    @patch("main_controller.Thread", _ImmediateThread)
    @patch("main_controller.export_display_state")
    def test_start_export_display_state_async_pauses_and_resumes_workers(self, export_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_db = MagicMock()
            display = self._build_display(db=fake_db)
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    historic_download_check_interval=19,
                ),
            )
            controller.remote_db_poll_thread = _AliveThread()
            controller.stop_historic_download_worker = MagicMock()
            controller.stop_remote_db_polling = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()
            controller.start_remote_db_polling = MagicMock()

            export_mock.return_value = {
                "ok": True,
                "package_path": f"{tmp_dir}\\display_state_20260326_120000.zip",
                "package_name": "display_state_20260326_120000.zip",
            }

            with patch("db.get_db_connection", return_value=fake_db):
                controller.start_export_display_state_async()

            controller.stop_historic_download_worker.assert_called_once_with()
            controller.stop_remote_db_polling.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmp_dir,
                check_interval=19,
            )
            controller.start_remote_db_polling.assert_called_once_with()
            export_mock.assert_called_once()
            self.assertFalse(display.sync_in_progress)
            self.assertEqual(display.sync_progress_title, "Exporting Dataset")
            self.assertEqual(display.sync_message, "Export completed: display_state_20260326_120000.zip")
            self.assertFalse(display.sync_message_is_error)
            fake_db.close.assert_called_once_with()

    @patch("main_controller.Thread", _ImmediateThread)
    @patch("main_controller.import_display_state")
    def test_start_import_display_state_async_reports_merge_summary(self, import_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_db = MagicMock()
            display = self._build_display(db=fake_db)
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    historic_download_check_interval=21,
                ),
            )
            controller.remote_db_poll_thread = _AliveThread()
            controller.stop_historic_download_worker = MagicMock()
            controller.stop_remote_db_polling = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()
            controller.start_remote_db_polling = MagicMock()

            import_mock.return_value = {
                "ok": True,
                "annotated": {"copied": 2, "skipped": 1},
                "historic": {"copied": 2, "skipped": 1},
                "db": {
                    "inserted": {
                        "img_results": 2,
                        "piece_result": 0,
                        "classified_images": 2,
                        "classified_image_defects": 1,
                        "piece_result_defects": 1,
                    },
                    "skipped": {
                        "img_results": 1,
                        "piece_result": 1,
                        "classified_images": 0,
                        "classified_image_defects": 0,
                        "piece_result_defects": 0,
                    },
                },
            }

            with patch("db.get_db_connection", return_value=fake_db):
                controller.start_import_display_state_async("C:\\tmp\\package.zip")

            import_mock.assert_called_once()
            self.assertEqual(import_mock.call_args.kwargs["package_path"], "C:\\tmp\\package.zip")
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.stop_remote_db_polling.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmp_dir,
                check_interval=21,
            )
            controller.start_remote_db_polling.assert_called_once_with()
            self.assertFalse(display.sync_in_progress)
            self.assertEqual(display.sync_progress_title, "Importing Dataset")
            self.assertEqual(
                display.sync_message,
                "Import completed: 4 files, 6 DB rows added, 4 duplicates skipped",
            )
            self.assertFalse(display.sync_message_is_error)
            fake_db.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)

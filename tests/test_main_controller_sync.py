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


class TestMainControllerSync(unittest.TestCase):
    def _build_display(self, db=None):
        display = MagicMock()
        display.db = db
        display.sftp_client = None
        display.sftp_credentials = None
        display.sync_in_progress = False
        display.reset_in_progress = False
        display.sync_progress = 0
        display.sync_stage = ""
        display.sync_message = ""
        display.sync_message_is_error = False
        display.sync_message_time = 0
        display.set_db_connection = MagicMock()
        return display

    @patch("main_controller.Thread", _ImmediateThread)
    def test_start_sync_async_stops_and_restarts_historic_download_workers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_db = MagicMock()
            display = self._build_display(db=fake_db)
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    historic_download_check_interval=17,
                ),
            )

            events = []
            visible_snapshot = ["img_b.png", "img_a.png"]
            controller.stop_historic_download_worker = MagicMock(
                side_effect=lambda: events.append("stop")
            )
            controller.start_historic_download_on_startup = MagicMock(
                side_effect=lambda *args, **kwargs: events.append("start")
            )
            controller._get_visible_historic_image_snapshot = MagicMock(
                return_value=visible_snapshot
            )
            controller.sync_images_by_status = MagicMock(
                side_effect=lambda **kwargs: events.append("sync") or {"ok": True, "rows_snapshot": []}
            )
            controller.save_classification_results = MagicMock(
                side_effect=lambda **kwargs: events.append("classify")
                or {
                    "ok": True,
                    "images": 3,
                    "files_copied": 3,
                    "stats_report_path": "reports/reporte_20260325_20260325_0900_1200.xlsx",
                }
            )
            controller.verify_sync_images_by_status = MagicMock(
                side_effect=lambda **kwargs: events.append("verify") or {"verified": True}
            )

            with patch("db.get_db_connection", return_value=fake_db):
                controller.start_sync_images_by_status_async()

            self.assertEqual(events, ["stop", "sync", "classify", "verify", "start"])
            self.assertEqual(
                controller.sync_images_by_status.call_args.kwargs["visible_images_snapshot"],
                visible_snapshot,
            )
            self.assertEqual(
                controller.save_classification_results.call_args.kwargs["visible_images_snapshot"],
                visible_snapshot,
            )
            self.assertTrue(
                controller.save_classification_results.call_args.kwargs["export_stats_report"]
            )
            self.assertEqual(
                controller.verify_sync_images_by_status.call_args.kwargs["visible_images_snapshot"],
                visible_snapshot,
            )
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmp_dir,
                check_interval=17,
            )
            self.assertFalse(display.sync_in_progress)
            self.assertEqual(
                display.sync_message,
                "Dataset saved: 3 images, 3 files copied. Report: reporte_20260325_20260325_0900_1200.xlsx",
            )
            self.assertFalse(display.sync_message_is_error)
            fake_db.close.assert_called_once_with()

    @patch("main_controller.Thread", _ImmediateThread)
    def test_start_sync_async_marks_report_warning_without_failing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_db = MagicMock()
            display = self._build_display(db=fake_db)
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
            )
            controller.stop_historic_download_worker = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()
            controller._get_visible_historic_image_snapshot = MagicMock(return_value=["img_a.png"])
            controller.sync_images_by_status = MagicMock(return_value={"ok": True, "rows_snapshot": []})
            controller.save_classification_results = MagicMock(
                return_value={
                    "ok": True,
                    "images": 2,
                    "files_copied": 2,
                    "stats_report_error": "Stats report requires a valid DB date range",
                }
            )
            controller.verify_sync_images_by_status = MagicMock(return_value={"verified": True})

            with patch("db.get_db_connection", return_value=fake_db):
                controller.start_sync_images_by_status_async()

            self.assertTrue(display.sync_message.startswith("Dataset saved: 2 images, 2 files copied. Report warning:"))
            self.assertTrue(display.sync_message_is_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)

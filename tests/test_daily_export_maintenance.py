import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from daily_export_maintenance import DailyExportMaintenance
from file_manager import FileManager


class _ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._alive = False

    def start(self):
        self._alive = True
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        finally:
            self._alive = False

    def is_alive(self):
        return self._alive


class TestDailyExportMaintenance(unittest.TestCase):
    def _build_runner(self, tmp_dir, now=None):
        display = SimpleNamespace(
            sync_in_progress=False,
            reset_in_progress=False,
            reset_progress=0,
            reset_progress_title="",
            reset_progress_helper_text="",
            reset_stage="",
            sync_message="",
            sync_message_is_error=False,
            sync_message_time=0,
        )
        config = SimpleNamespace(
            daily_maintenance_enabled=True,
            daily_maintenance_hour=5,
            daily_maintenance_minute=40,
            daily_maintenance_min_free_bytes=0,
            daily_maintenance_retry_interval_sec=0,
            daily_maintenance_exports_dir=str(Path(tmp_dir) / "exports"),
        )
        db = MagicMock()
        db.truncate_app_tables.return_value = 4
        logger = MagicMock()
        controller = SimpleNamespace(
            display=display,
            config=config,
            file_manager=FileManager(),
            logger=logger,
            dataset_transfer_active=False,
            _pause_dataset_background_workers=MagicMock(return_value={"remote_db_was_running": False}),
            _resume_dataset_background_workers=MagicMock(),
            _set_reset_progress=MagicMock(),
            perform_reset=MagicMock(return_value={"ok": True}),
        )
        runner = DailyExportMaintenance(
            controller,
            now_fn=lambda: now or datetime(2026, 5, 19, 5, 40, 0),
        )
        return runner, controller, db

    @patch("daily_export_maintenance.Thread", _ImmediateThread)
    @patch("daily_export_maintenance.estimate_display_state_export_size")
    @patch("daily_export_maintenance.export_display_state")
    def test_tick_runs_after_scheduled_time_in_order(self, export_mock, estimate_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, controller, db = self._build_runner(tmp_dir)
            calls = []
            estimate_mock.side_effect = lambda *args, **kwargs: calls.append("estimate") or {
                "ok": True,
                "required_bytes": 1,
            }
            export_mock.side_effect = lambda *args, **kwargs: calls.append("export") or {
                "ok": True,
                "package_path": str(Path(tmp_dir) / "exports" / "display_state_20260519_054000"),
                "package_name": "display_state_20260519_054000",
            }
            controller.perform_reset.side_effect = lambda *args, **kwargs: calls.append("reset") or {
                "ok": True
            }
            db.truncate_app_tables.side_effect = lambda: calls.append("truncate") or 4

            with patch("db.get_db_connection", return_value=db), patch(
                "daily_export_maintenance.shutil.disk_usage",
                return_value=SimpleNamespace(free=10_000),
            ):
                self.assertTrue(runner.tick())

            self.assertEqual(calls, ["estimate", "export", "reset", "truncate"])
            self.assertEqual(controller.display.sync_message, "Daily export/reset completed: display_state_20260519_054000")
            self.assertFalse(controller.display.sync_message_is_error)
            db.close.assert_called_once_with()

    @patch("daily_export_maintenance.Thread", _ImmediateThread)
    @patch("daily_export_maintenance.export_display_state")
    def test_tick_does_not_repeat_after_success_today(self, export_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, _controller, _db = self._build_runner(tmp_dir)
            runner.exports_dir.mkdir(parents=True)
            runner._save_state({"last_success_date": "2026-05-19"})

            self.assertFalse(runner.tick())
            export_mock.assert_not_called()

    @patch("daily_export_maintenance.Thread", _ImmediateThread)
    @patch("daily_export_maintenance.export_display_state")
    def test_start_async_skips_when_display_is_busy(self, export_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, controller, _db = self._build_runner(tmp_dir)
            controller.display.reset_in_progress = True

            self.assertFalse(runner.start_async())
            export_mock.assert_not_called()

    def test_storage_check_passes_when_space_is_enough(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, _controller, _db = self._build_runner(tmp_dir)

            with patch(
                "daily_export_maintenance.shutil.disk_usage",
                return_value=SimpleNamespace(free=10_000),
            ):
                result = runner._check_export_storage(450)

            self.assertTrue(result["ok"])
            self.assertEqual(result["available_free_bytes"], 10_000)

    def test_storage_check_aborts_without_deleting_exports_when_space_is_low(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, _controller, _db = self._build_runner(tmp_dir)
            exports_dir = runner.exports_dir
            exports_dir.mkdir(parents=True)
            old_export = exports_dir / "display_state_20260517_054000"
            old_export.mkdir()
            (old_export / "payload.bin").write_bytes(b"a" * 100)

            with patch(
                "daily_export_maintenance.shutil.disk_usage",
                return_value=SimpleNamespace(free=0),
            ):
                result = runner._check_export_storage(500)

            self.assertFalse(result["ok"])
            self.assertTrue(old_export.exists())

    @patch("daily_export_maintenance.Thread", _ImmediateThread)
    @patch("daily_export_maintenance.estimate_display_state_export_size")
    @patch("daily_export_maintenance.export_display_state")
    def test_low_storage_does_not_export_reset_truncate_or_delete(self, export_mock, estimate_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, controller, db = self._build_runner(tmp_dir)
            exports_dir = runner.exports_dir
            exports_dir.mkdir(parents=True)
            old_export = exports_dir / "display_state_20260517_054000"
            old_export.mkdir()
            (old_export / "payload.bin").write_bytes(b"a" * 100)
            estimate_mock.return_value = {"ok": True, "required_bytes": 500}

            with patch("db.get_db_connection", return_value=db), patch(
                "daily_export_maintenance.shutil.disk_usage",
                return_value=SimpleNamespace(free=0),
            ):
                self.assertTrue(runner.tick())

            self.assertTrue(old_export.exists())
            export_mock.assert_not_called()
            controller.perform_reset.assert_not_called()
            db.truncate_app_tables.assert_not_called()
            self.assertTrue(controller.display.sync_message_is_error)

    @patch("daily_export_maintenance.Thread", _ImmediateThread)
    @patch("daily_export_maintenance.estimate_display_state_export_size")
    @patch("daily_export_maintenance.export_display_state")
    def test_export_failure_does_not_reset_or_truncate(self, export_mock, estimate_mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner, controller, db = self._build_runner(tmp_dir)
            estimate_mock.return_value = {"ok": True, "required_bytes": 1}
            export_mock.return_value = {"ok": False, "error": "boom"}

            with patch("db.get_db_connection", return_value=db), patch(
                "daily_export_maintenance.shutil.disk_usage",
                return_value=SimpleNamespace(free=10_000),
            ):
                self.assertTrue(runner.tick())

            controller.perform_reset.assert_not_called()
            db.truncate_app_tables.assert_not_called()
            self.assertTrue(controller.display.sync_message_is_error)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from file_manager import FileManager
from main_controller import ControllerConfig, MainController


class _ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


class TestMainControllerRebuild(unittest.TestCase):
    def _build_display(self, db):
        display = MagicMock()
        display.db = db
        return display

    def test_rebuild_preserves_classified_and_final_classification_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            historic_dir = tmp_path / "historic"
            historic_dir.mkdir()
            (historic_dir / "JSN001_side_OK.png").write_bytes(b"test-image")

            sync_base_dir = tmp_path / "classified"
            (sync_base_dir / "side_ok").mkdir(parents=True)
            (sync_base_dir / "side_ok" / "old_side.png").write_bytes(b"old")
            final_classification_dir = tmp_path / "final_classification"
            (final_classification_dir / "Side_P").mkdir(parents=True)
            (final_classification_dir / "Side_P" / "old_side.png").write_bytes(b"old")

            db = MagicMock()
            db.truncate_app_tables.return_value = 3
            db.fetch.return_value = [{"cnt": 1}]

            controller = MainController(
                display=self._build_display(db),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()
            controller._backfill_piece_result = MagicMock()
            controller._clear_final_classification_dir = MagicMock(return_value=0)
            controller._clear_sync_images_base_dir = MagicMock(return_value=0)
            controller._invalidate_dataset_runtime_state = MagicMock()
            controller.enter_historic_mode = MagicMock()
            controller.exit_historic_mode = MagicMock()

            progress_updates = []

            with patch("main_controller.SYNC_IMAGES_BASE_DIR", sync_base_dir), patch(
                "main_controller.FINAL_CLASSIFICATION_DIR", final_classification_dir
            ):
                result = controller.perform_rebuild_db_from_historic(
                    progress_callback=lambda done, total, stage: progress_updates.append(
                        (done, total, stage)
                    )
                )

            self.assertTrue(result["ok"])
            self.assertNotIn((3, 4, "Clearing final classification"), progress_updates)
            self.assertNotIn((4, 4, "Clearing classified folders"), progress_updates)
            controller._clear_final_classification_dir.assert_not_called()
            controller._clear_sync_images_base_dir.assert_not_called()
            self.assertTrue((sync_base_dir / "side_ok" / "old_side.png").exists())
            self.assertTrue((final_classification_dir / "Side_P" / "old_side.png").exists())

    def test_async_rebuild_pauses_and_resumes_remote_db_worker(self):
        db = MagicMock()
        display = self._build_display(db)
        display.reset_in_progress = False
        display.sync_in_progress = False
        controller = MainController(
            display=display,
            config=ControllerConfig(),
            file_manager=FileManager(),
        )
        controller._pause_remote_db_background_worker = MagicMock(return_value=True)
        controller._resume_remote_db_background_worker = MagicMock()
        controller.perform_rebuild_db_from_historic = MagicMock(return_value={"ok": True})

        with patch("main_controller.Thread", _ImmediateThread), patch(
            "db.get_db_connection",
            return_value=db,
        ):
            controller.start_rebuild_db_from_historic_async()

        controller._pause_remote_db_background_worker.assert_called_once_with()
        controller._resume_remote_db_background_worker.assert_called_once_with(True)
        controller.perform_rebuild_db_from_historic.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from file_manager import FileManager
from main_controller import MainController
from paths_config import STATUS_SYNC_DIRS


class TestMainControllerRebuild(unittest.TestCase):
    def _build_display(self, db):
        display = MagicMock()
        display.db = db
        return display

    def test_rebuild_clears_sync_images_base_dir_and_keeps_expected_empty_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            annotated_dir.mkdir()
            (annotated_dir / "JSN001_side_OK.png").write_bytes(b"test-image")

            sync_base_dir = tmp_path / "classified"
            (sync_base_dir / "side_ok").mkdir(parents=True)
            (sync_base_dir / "side_ok" / "old_side.png").write_bytes(b"old")
            (sync_base_dir / "front_nok").mkdir(parents=True)
            (sync_base_dir / "front_nok" / "nested").mkdir()
            (sync_base_dir / "front_nok" / "nested" / "old_front.png").write_bytes(b"old")
            (sync_base_dir / "diag_ok").mkdir(parents=True)
            (sync_base_dir / "unexpected_folder").mkdir(parents=True)
            (sync_base_dir / "unexpected_folder" / "junk.txt").write_text("junk", encoding="utf-8")
            (sync_base_dir / "stray_file.txt").write_text("junk", encoding="utf-8")

            db = MagicMock()
            db.truncate_app_tables.return_value = 3
            db.fetch.return_value = [{"cnt": 1}]

            controller = MainController(
                display=self._build_display(db),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()
            controller._backfill_piece_result = MagicMock()
            controller._clear_final_classification_dir = MagicMock(return_value=0)
            controller._invalidate_dataset_runtime_state = MagicMock()
            controller.enter_historic_mode = MagicMock()
            controller.exit_historic_mode = MagicMock()

            progress_updates = []

            with patch("main_controller.ANNOTATED_LOCAL_DIR", annotated_dir), patch(
                "main_controller.SYNC_IMAGES_BASE_DIR", sync_base_dir
            ):
                result = controller.perform_rebuild_db_from_historic(
                    progress_callback=lambda done, total, stage: progress_updates.append(
                        (done, total, stage)
                    )
                )

            self.assertTrue(result["ok"])
            self.assertIn(
                (5, 6, "Clearing classified folders"),
                progress_updates,
            )

            expected_folders = {
                folder_name
                for position_dirs in STATUS_SYNC_DIRS.values()
                for folder_name in position_dirs.values()
            }

            self.assertEqual(set(p.name for p in sync_base_dir.iterdir()), expected_folders)
            for folder_name in expected_folders:
                folder_path = sync_base_dir / folder_name
                self.assertTrue(folder_path.is_dir(), f"{folder_name} should exist")
                self.assertEqual(list(folder_path.iterdir()), [], f"{folder_name} should be empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)

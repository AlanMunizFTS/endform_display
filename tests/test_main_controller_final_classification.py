import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from file_manager import FileManager
from main_controller import ControllerConfig, MainController


class _ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class TestMainControllerFinalClassification(unittest.TestCase):
    def _build_display(self, db=None):
        display = MagicMock()
        display.db = db
        display.historic_images = []
        display.historic_offset = 0
        display.historic_mode = False
        display.historic_db_registered = False
        display.historic_index_rescan_interval = 1.0
        display.temp_results = {}
        display.available_jsns = []
        display.filtered_suggestions = []
        display.search_jsn = ""
        display.search_active = False
        display.selected_suggestion_idx = -1
        display.show_reset_confirm = False
        display.show_delete_confirm = False
        display.show_rebuild_confirm = False
        display.show_piece_date_dialog = False
        display._db_registered_images = set()
        display._db_result_cache = {}
        display._image_cache = {}
        display._historic_index_cache = None
        display._historic_jsn_cache = []
        display._historic_index_mtime = None
        display._historic_index_last_scan = 0.0
        display.sftp_client = None
        display.set_db_connection = MagicMock()
        return display

    @patch("main_controller.Thread", _ImmediateThread)
    def test_new_annotated_images_do_not_auto_save_final_classification(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            annotated_dir.mkdir()
            (annotated_dir / "118610000000000000001_Cam1_Side1_OK.png").write_bytes(b"annotated")

            fake_db = MagicMock()
            controller = MainController(
                display=self._build_display(db=fake_db),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller.last_historic_mtime = -1
            controller._register_local_images_in_db = MagicMock()
            controller._backfill_piece_result = MagicMock()
            controller.save_classification_results = MagicMock()

            with patch("db.get_db_connection", return_value=fake_db):
                controller._check_and_register_new_historic_images()

            controller._register_local_images_in_db.assert_any_call(
                str(annotated_dir),
                db_client=fake_db,
            )
            controller._backfill_piece_result.assert_any_call(db_client=fake_db)
            controller.save_classification_results.assert_not_called()
            fake_db.close.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from file_manager import FileManager
from main_controller import ControllerConfig, MainController
from paths_config import FINAL_CLASSIFICATION_DIRS, STATUS_SYNC_DIRS


class TestMainControllerAnnotatedHistoric(unittest.TestCase):
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

    def test_load_historic_index_uses_annotated_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()

            annotated_names = [
                "118610000000000000001_Cam2_Diag1_OK.png",
                "118610000000000000001_Cam1_Side1_OK.png",
                "118610000000000000002_Cam3_Front_NOK.png",
            ]
            for name in annotated_names:
                (annotated_dir / name).write_bytes(b"annotated")

            (historic_dir / "118619999999999999999_Cam1_Side1_OK.png").write_bytes(b"historic")

            controller = MainController(
                display=self._build_display(),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )

            historic_index = controller._load_historic_index(force_rescan=True)

            self.assertEqual(
                historic_index,
                [
                    [
                        "118610000000000000002_Cam3_Front_NOK.png",
                    ],
                    [
                        "118610000000000000001_Cam1_Side1_OK.png",
                        "118610000000000000001_Cam2_Diag1_OK.png",
                    ],
                ],
            )

    def test_download_historic_batch_returns_annotated_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            annotated_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            img_path = annotated_dir / img_name
            img_path.write_bytes(b"annotated")

            display = self._build_display()
            display.historic_images = [[img_name]]
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(result, [str(img_path)])
            controller._register_local_images_in_db.assert_called_once_with(
                str(annotated_dir),
                image_names=[img_name],
            )

    def test_save_classification_results_reports_missing_historic_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            annotated_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            (annotated_dir / img_name).write_bytes(b"annotated")

            db = MagicMock()
            db.fetch.return_value = [{"img_name": img_name, "result": "OK"}]

            controller = MainController(
                display=self._build_display(db=db),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )

            with patch("main_controller.FINAL_CLASSIFICATION_DIR", tmp_path / "final_classification"):
                result = controller.save_classification_results(db_client=db)

            self.assertTrue(result["ok"])
            self.assertIn("classification_folder_errors", result)
            self.assertIn(f"Source missing: {img_name}", result["classification_folder_errors"])

    def test_save_classification_results_returns_stats_report_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            historic_dir = tmp_path / "historic"
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            (historic_dir / img_name).write_bytes(b"historic")

            db = MagicMock()
            db.fetch.return_value = [{"img_name": img_name, "result": "OK"}]

            controller = MainController(
                display=self._build_display(db=db),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )

            report_path = tmp_path / "reports" / "reporte_20260325_20260325_0900_1200.xlsx"
            with patch("main_controller.FINAL_CLASSIFICATION_DIR", tmp_path / "final_classification"), patch(
                "report_exporter.export_stats_report",
                return_value=str(report_path),
            ) as mock_export:
                result = controller.save_classification_results(
                    db_client=db,
                    historic_dir=str(historic_dir),
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["stats_report_path"], str(report_path))
            mock_export.assert_called_once()

    def test_perform_delete_current_piece_removes_annotated_and_historic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()

            target_name = "118610000000000000001_Cam1_Side1_OK.png"
            survivor_name = "118610000000000000002_Cam1_Side1_OK.png"
            for base_dir in (annotated_dir, historic_dir):
                (base_dir / target_name).write_bytes(b"target")
                (base_dir / survivor_name).write_bytes(b"survivor")

            db = MagicMock()
            db.execute.return_value = 1
            display = self._build_display(db=db)
            display.historic_images = [[target_name], [survivor_name]]

            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller.enter_historic_mode = MagicMock()
            controller.exit_historic_mode = MagicMock()
            controller._show_no_images_dialog = MagicMock()

            controller.perform_delete_current_piece()

            self.assertFalse((annotated_dir / target_name).exists())
            self.assertFalse((historic_dir / target_name).exists())
            self.assertTrue((annotated_dir / survivor_name).exists())
            self.assertTrue((historic_dir / survivor_name).exists())
            controller.enter_historic_mode.assert_called_once()

    def test_perform_reset_clears_classified_and_final_dirs_and_keeps_live_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()
            (annotated_dir / "annotated.png").write_bytes(b"annotated")
            (historic_dir / "historic.png").write_bytes(b"historic")
            live_root_file = tmp_path / "live_root.png"
            live_root_file.write_bytes(b"live")

            sync_base_dir = tmp_path / "classified"
            (sync_base_dir / "side_ok").mkdir(parents=True)
            (sync_base_dir / "side_ok" / "old_side.png").write_bytes(b"old")
            (sync_base_dir / "unexpected_folder").mkdir(parents=True)
            (sync_base_dir / "unexpected_folder" / "junk.txt").write_text("junk", encoding="utf-8")

            final_classification_dir = tmp_path / "final_classification"
            (final_classification_dir / "Side_P").mkdir(parents=True)
            (final_classification_dir / "Side_P" / "old_side.png").write_bytes(b"old")
            (final_classification_dir / "unexpected_folder").mkdir(parents=True)
            (final_classification_dir / "unexpected_folder" / "junk.txt").write_text(
                "junk",
                encoding="utf-8",
            )

            db = MagicMock()
            db.execute.return_value = 1
            display = self._build_display(db=db)
            display.sftp_client = object()

            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    remote_hist_dir="/remote/historic",
                    remote_annotated_dir="/remote/annotated",
                ),
                file_manager=FileManager(),
            )
            controller.file_manager.sftp_chdir = MagicMock()
            controller.file_manager.sftp_listdir = MagicMock(
                side_effect=[["remote_historic.png"], ["remote_annotated.png"]]
            )
            controller.file_manager.sftp_remove = MagicMock()

            with patch("main_controller.SYNC_IMAGES_BASE_DIR", sync_base_dir), patch(
                "main_controller.FINAL_CLASSIFICATION_DIR", final_classification_dir
            ):
                result = controller.perform_reset(db_client=db)

            self.assertTrue(result["ok"])
            self.assertEqual(list(annotated_dir.iterdir()), [])
            self.assertEqual(list(historic_dir.iterdir()), [])
            self.assertTrue(live_root_file.exists())

            expected_sync_folders = {
                folder_name
                for position_dirs in STATUS_SYNC_DIRS.values()
                for folder_name in position_dirs.values()
            }
            self.assertEqual(set(p.name for p in sync_base_dir.iterdir()), expected_sync_folders)
            for folder_name in expected_sync_folders:
                self.assertEqual(list((sync_base_dir / folder_name).iterdir()), [])

            expected_final_folders = {
                folder_name
                for position_dirs in FINAL_CLASSIFICATION_DIRS.values()
                for folder_name in position_dirs.values()
            }
            self.assertEqual(
                set(p.name for p in final_classification_dir.iterdir()),
                expected_final_folders,
            )
            for folder_name in expected_final_folders:
                self.assertEqual(list((final_classification_dir / folder_name).iterdir()), [])

            controller.file_manager.sftp_remove.assert_any_call(
                display.sftp_client,
                "/remote/annotated/remote_annotated.png",
            )
            controller.file_manager.sftp_remove.assert_any_call(
                display.sftp_client,
                "/remote/historic/remote_historic.png",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

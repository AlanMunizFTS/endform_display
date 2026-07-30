import tempfile
import unittest
from pathlib import Path
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
                "package_path": f"{tmp_dir}\\display_state_20260326_120000",
                "package_name": "display_state_20260326_120000",
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
            self.assertEqual(display.sync_message, "Export completed: display_state_20260326_120000")
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
                "annotated": {"copied": 0, "skipped": 0},
                "historic": {"copied": 2, "skipped": 1},
                "db": {
                    "inserted": {
                        "img_results": 2,
                        "piece_result": 0,
                        "classified_images": 2,
                        "classified_image_defects": 1,
                        "model_results": 2,
                        "piece_result_defects": 1,
                    },
                    "skipped": {
                        "img_results": 1,
                        "piece_result": 1,
                        "classified_images": 0,
                        "classified_image_defects": 0,
                        "model_results": 0,
                        "piece_result_defects": 0,
                    },
                },
            }

            with patch("db.get_db_connection", return_value=fake_db):
                controller.start_import_display_state_async("C:\\tmp\\display_state_20260326_120000")

            import_mock.assert_called_once()
            self.assertEqual(
                import_mock.call_args.kwargs["package_path"],
                "C:\\tmp\\display_state_20260326_120000",
            )
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
                "Import completed: 2 files, 8 DB rows added, 3 duplicates skipped",
            )
            self.assertFalse(display.sync_message_is_error)
            fake_db.close.assert_called_once_with()

    @patch("main_controller.Thread", _ImmediateThread)
    @patch("main_controller.export_piece_stats_dataset")
    def test_start_export_piece_stats_dataset_async_reports_output_folder(self, export_mock):
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
            controller.remote_db_poll_thread = _AliveThread()
            controller.stop_historic_download_worker = MagicMock()
            controller.stop_remote_db_polling = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()
            controller.start_remote_db_polling = MagicMock()

            export_mock.return_value = {
                "ok": True,
                "dataset_name": "dataset_20260424_101500",
                "output_path": f"{tmp_dir}\\datasets\\dataset_20260424_101500",
                "matched_images": 8,
                "copied_files": 11,
                "missing_count": 0,
            }

            with patch("db.get_db_connection", return_value=fake_db):
                controller.start_export_piece_stats_dataset_async(
                    filters={
                        "results": ["OK", "NOK"],
                        "angles": ["side"],
                        "class_names": ["All"],
                    }
                )

            export_mock.assert_called_once()
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.stop_remote_db_polling.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmp_dir,
                check_interval=17,
            )
            controller.start_remote_db_polling.assert_called_once_with()
            self.assertFalse(display.sync_in_progress)
            self.assertEqual(display.sync_progress_title, "Exporting Piece Stats Dataset")
            self.assertEqual(
                display.sync_message,
                "Dataset export completed: dataset_20260424_101500 (8 images, 11 copies)",
            )
            self.assertFalse(display.sync_message_is_error)
            fake_db.close.assert_called_once_with()

    @patch("main_controller.Thread", _ImmediateThread)
    def test_start_export_piece_stats_report_async_exports_combined_workbook(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_db = MagicMock()
            display = self._build_display(db=fake_db)
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    historic_download_check_interval=15,
                ),
            )
            controller.remote_db_poll_thread = _AliveThread()
            controller.stop_historic_download_worker = MagicMock()
            controller.stop_remote_db_polling = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()
            controller.start_remote_db_polling = MagicMock()

            with patch("db.get_db_connection", return_value=fake_db):
                with patch(
                    "report_exporter.export_combined_traceability_report",
                    return_value=f"{tmp_dir}\\reports\\desglose_ok_nok_20260424_101500.xlsx",
                ) as export_mock:
                    controller.start_export_piece_stats_report_async()

            export_mock.assert_called_once()
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.stop_remote_db_polling.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmp_dir,
                check_interval=15,
            )
            controller.start_remote_db_polling.assert_called_once_with()
            self.assertFalse(display.sync_in_progress)
            self.assertEqual(display.sync_progress_title, "Exporting Excel Reports")
            self.assertEqual(
                display.sync_message,
                "Excel report exported: desglose_ok_nok_20260424_101500.xlsx",
            )
            self.assertFalse(display.sync_message_is_error)
            fake_db.close.assert_called_once_with()

    @patch("main_controller.Thread", _ImmediateThread)
    def test_start_export_historic_image_report_async_exports_workbook(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            display = self._build_display(db=MagicMock())
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    historic_download_check_interval=15,
                ),
            )
            controller.remote_db_poll_thread = _AliveThread()
            controller.stop_historic_download_worker = MagicMock()
            controller.stop_remote_db_polling = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()
            controller.start_remote_db_polling = MagicMock()

            with patch(
                "report_exporter.export_historic_image_table_report",
                return_value=f"{tmp_dir}\\reports\\reporte_imagenes_historico_20260514_093000.xlsx",
            ) as export_mock:
                controller.start_export_historic_image_report_async(
                    endform_type="mush",
                    class_name="wrinkle",
                    defect_class="wrinkle",
                    angle="diag",
                )

            export_mock.assert_called_once()
            self.assertEqual(export_mock.call_args.kwargs["endform_type"], "mush")
            self.assertEqual(export_mock.call_args.kwargs["class_name"], "wrinkle")
            self.assertEqual(export_mock.call_args.kwargs["defect_class"], "wrinkle")
            self.assertEqual(export_mock.call_args.kwargs["angle"], "diag")
            self.assertEqual(export_mock.call_args.kwargs["pieces_per_group"], 4)
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.stop_remote_db_polling.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmp_dir,
                check_interval=15,
            )
            controller.start_remote_db_polling.assert_called_once_with()
            self.assertFalse(display.sync_in_progress)
            self.assertEqual(display.sync_progress_title, "Exporting Image Report")
            self.assertEqual(
                display.sync_message,
                "Image report exported: reporte_imagenes_historico_20260514_093000.xlsx",
            )
            self.assertFalse(display.sync_message_is_error)

    def test_check_and_register_new_historic_images_skips_during_dataset_transfer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotated_dir = Path(tmp_dir) / "annotated"
            annotated_dir.mkdir()
            (annotated_dir / "11861_test_side_OK.png").write_bytes(b"image")

            display = self._build_display(db=MagicMock())
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
            )
            controller.dataset_transfer_active = True
            controller.file_manager.getmtime = MagicMock(return_value=123.0)
            controller._register_local_images_in_db = MagicMock()
            controller._backfill_piece_result = MagicMock()
            controller.save_classification_results = MagicMock()

            controller._check_and_register_new_historic_images()

            self.assertFalse(getattr(controller, "_register_worker_running", False))
            controller._register_local_images_in_db.assert_not_called()
            controller._backfill_piece_result.assert_not_called()
            controller.save_classification_results.assert_not_called()

    @patch("main_controller.Thread", _ImmediateThread)
    def test_check_and_register_new_historic_images_does_not_export_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotated_dir = Path(tmp_dir) / "annotated"
            annotated_dir.mkdir()
            (annotated_dir / "11861_test_side_OK.png").write_bytes(b"image")

            fake_db = MagicMock()
            display = self._build_display(db=MagicMock())
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
            )
            controller.last_historic_mtime = 100.0
            controller.file_manager.getmtime = MagicMock(return_value=200.0)
            controller._register_local_images_in_db = MagicMock()
            controller._backfill_piece_result = MagicMock()
            controller.save_classification_results = MagicMock(return_value={"ok": True})

            with patch("db.get_db_connection", return_value=fake_db):
                controller._check_and_register_new_historic_images()

            controller.save_classification_results.assert_not_called()
            fake_db.close.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

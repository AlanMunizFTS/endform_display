import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

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

    def test_load_historic_index_uses_historic_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()

            historic_names = [
                "118610000000000000001_Cam2_Diag1_OK.png",
                "118610000000000000001_Cam1_Side1_OK.png",
                "118610000000000000002_Cam3_Front_NOK.png",
            ]
            for name in historic_names:
                (historic_dir / name).write_bytes(b"historic")

            (annotated_dir / "118619999999999999999_Cam1_Side1_OK.png").write_bytes(b"annotated")

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

    def test_download_historic_batch_returns_historic_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            historic_dir = tmp_path / "historic"
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            img_path = historic_dir / img_name
            img_path.write_bytes(b"historic")

            display = self._build_display()
            display.historic_images = [[img_name]]
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["img_name"], img_name)
            self.assertEqual(result[0]["path"], str(img_path))
            self.assertEqual(result[0]["status"], "ready")
            self.assertEqual(result[0]["source"], "historic")
            controller._register_local_images_in_db.assert_called_once_with(
                str(historic_dir),
                image_names=[img_name],
            )

    def test_download_historic_batch_uses_annotated_fallback_without_db_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            annotated_path = annotated_dir / img_name
            historic_path = historic_dir / img_name
            annotated_path.write_bytes(b"annotated")
            historic_path.write_bytes(b"historic")

            display = self._build_display()
            display.historic_images = [[img_name]]
            logger = MagicMock()
            controller = MainController(
                display=display,
                logger=logger,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["img_name"], img_name)
            self.assertEqual(result[0]["path"], str(annotated_path))
            self.assertEqual(result[0]["status"], "ready")
            self.assertEqual(result[0]["source"], "annotated_fallback")
            self.assertTrue(
                any(f"{img_name}=annotated_fallback" in call.args[0] for call in logger.info.call_args_list)
            )

    def test_download_historic_batch_prefers_db_coordinates_over_annotated_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            annotated_path = annotated_dir / img_name
            historic_path = historic_dir / img_name
            annotated_path.write_bytes(b"annotated")
            historic_path.write_bytes(b"historic")

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "scratch",
                    "confidence": 0.91,
                    "model_name": "model",
                    "geometry_type": "bbox",
                    "coordinates": [10, 20, 100, 120],
                    "image_width": 360,
                    "image_height": 360,
                }
            ]
            display = self._build_display(db=db)
            display.historic_images = [[img_name]]
            logger = MagicMock()
            controller = MainController(
                display=display,
                logger=logger,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["img_name"], img_name)
            self.assertEqual(result[0]["path"], str(historic_path))
            self.assertEqual(result[0]["status"], "loading")
            self.assertEqual(result[0]["source"], "db_coordinates+historic")
            self.assertTrue(
                any(f"{img_name}=db_coordinates+historic" in call.args[0] for call in logger.info.call_args_list)
            )

    def test_download_historic_batch_does_not_use_annotated_when_coordinates_need_historic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            (annotated_dir / img_name).write_bytes(b"annotated")

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "scratch",
                    "confidence": 0.91,
                    "model_name": "model",
                    "geometry_type": "bbox",
                    "coordinates": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
                    "image_width": 360,
                    "image_height": 360,
                }
            ]
            display = self._build_display(db=db)
            display.historic_images = [[img_name]]
            logger = MagicMock()
            controller = MainController(
                display=display,
                logger=logger,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["img_name"], img_name)
            self.assertEqual(result[0]["status"], "missing")
            self.assertEqual(result[0]["source"], "missing_historic_with_coordinates")
            self.assertTrue(
                any("Coordinates exist but historic image is missing" in call.args[0] for call in logger.warn.call_args_list)
            )

    def test_download_historic_batch_ignores_classification_rows_for_overlay(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            annotated_path = annotated_dir / img_name
            annotated_path.write_bytes(b"annotated")
            (historic_dir / img_name).write_bytes(b"historic")

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "NOK",
                    "confidence": 0.91,
                    "model_name": "model",
                    "geometry_type": "classification",
                    "coordinates": None,
                    "image_width": 360,
                    "image_height": 360,
                }
            ]
            display = self._build_display(db=db)
            display.historic_images = [[img_name]]
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(result[0]["path"], str(annotated_path))
            self.assertEqual(result[0]["status"], "ready")
            self.assertEqual(result[0]["source"], "annotated_fallback")

    def test_download_historic_batch_finishes_overlay_worker_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            historic_dir = tmp_path / "historic"
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            historic_path = historic_dir / img_name
            cv2.imwrite(str(historic_path), np.zeros((40, 40, 3), dtype=np.uint8))

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "scratch",
                    "confidence": 0.91,
                    "model_name": "model",
                    "geometry_type": "bbox",
                    "coordinates": {"x1": 1, "y1": 2, "x2": 20, "y2": 22},
                    "image_width": 40,
                    "image_height": 40,
                }
            ]
            display = self._build_display(db=db)
            display.DEFAULT_TILE_SIZE = 20
            display.historic_images = [[img_name]]
            display._draw_model_overlays = MagicMock(side_effect=lambda img, *_args: img + 1)
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            first_result = controller.download_historic_batch(tmp_dir)
            self.assertEqual(first_result[0]["status"], "loading")
            controller.historic_render_worker_thread.join(timeout=2)

            second_result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(second_result[0]["status"], "ready")
            self.assertEqual(second_result[0]["source"], "db_coordinates+historic")
            self.assertIn("prepared_image", second_result[0])
            self.assertGreater(int(second_result[0]["prepared_image"].sum()), 0)
            self.assertEqual(display._draw_model_overlays.call_count, 1)

    def test_download_historic_batch_falls_back_to_annotated_when_overlay_render_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            annotated_dir = tmp_path / "annotated"
            historic_dir = tmp_path / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            annotated_path = annotated_dir / img_name
            annotated_path.write_bytes(b"annotated")
            cv2.imwrite(str(historic_dir / img_name), np.zeros((40, 40, 3), dtype=np.uint8))

            db = MagicMock()
            db.fetch.return_value = [
                {
                    "img_name": img_name,
                    "class_name": "scratch",
                    "confidence": 0.91,
                    "model_name": "model",
                    "geometry_type": "bbox",
                    "coordinates": {"x1": 1, "y1": 2, "x2": 20, "y2": 22},
                    "image_width": 40,
                    "image_height": 40,
                }
            ]
            display = self._build_display(db=db)
            display.DEFAULT_TILE_SIZE = 20
            display.historic_images = [[img_name]]
            display._draw_model_overlays = MagicMock(side_effect=RuntimeError("boom"))
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller._register_local_images_in_db = MagicMock()

            first_result = controller.download_historic_batch(tmp_dir)
            self.assertEqual(first_result[0]["source"], "db_coordinates+historic")
            self.assertEqual(first_result[0]["status"], "loading")
            controller.historic_render_worker_thread.join(timeout=2)

            second_result = controller.download_historic_batch(tmp_dir)

            self.assertEqual(second_result[0]["path"], str(annotated_path))
            self.assertEqual(second_result[0]["status"], "ready")
            self.assertEqual(second_result[0]["source"], "annotated_fallback")
            self.assertEqual(second_result[0]["error"], "boom")

    def test_stale_historic_overlay_worker_result_is_ignored_after_navigation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            historic_dir = tmp_path / "historic"
            historic_dir.mkdir()
            img_name = "118610000000000000001_Cam1_Side1_OK.png"
            historic_path = historic_dir / img_name
            cv2.imwrite(str(historic_path), np.zeros((40, 40, 3), dtype=np.uint8))

            display = self._build_display()
            display.DEFAULT_TILE_SIZE = 20
            display._draw_model_overlays = MagicMock(side_effect=lambda img, *_args: img + 1)
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller.historic_render_generation_id = 2
            controller.historic_render_items = {
                img_name: {
                    "img_name": img_name,
                    "status": "loading",
                    "source": "db_coordinates+historic",
                    "path": str(historic_path),
                }
            }

            controller._prepare_historic_render_worker(
                1,
                [
                    {
                        "img_name": img_name,
                        "path": str(historic_path),
                        "overlays": [
                            {
                                "class_name": "scratch",
                                "confidence": 0.91,
                                "geometry_type": "bbox",
                                "coordinates": {"x1": 1, "y1": 2, "x2": 20, "y2": 22},
                                "image_width": 40,
                                "image_height": 40,
                            }
                        ],
                        "cache_key": ("stale",),
                        "tile_size": 20,
                    }
                ],
            )

            self.assertEqual(controller.historic_render_items[img_name]["status"], "loading")
            display._draw_model_overlays.assert_not_called()

    def test_exit_historic_mode_clears_tile_items_from_display_paths(self):
        display = self._build_display()
        display.image_paths = [
            {
                "img_name": "118610000000000000001_Cam1_Side1_OK.png",
                "status": "loading",
                "source": "db_coordinates+historic",
            }
        ]
        display.historic_mode = True
        display.historic_images = [["118610000000000000001_Cam1_Side1_OK.png"]]
        controller = MainController(
            display=display,
            config=ControllerConfig(),
            file_manager=FileManager(),
        )

        controller.exit_historic_mode()

        self.assertFalse(display.historic_mode)
        self.assertEqual(display.image_paths, [])

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

    def test_save_classification_results_skips_stats_report_by_default(self):
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

            with patch("main_controller.FINAL_CLASSIFICATION_DIR", tmp_path / "final_classification"), patch(
                "report_exporter.export_stats_report"
            ) as mock_export:
                result = controller.save_classification_results(
                    db_client=db,
                    historic_dir=str(historic_dir),
                )

            self.assertTrue(result["ok"])
            self.assertNotIn("stats_report_path", result)
            mock_export.assert_not_called()

    def test_save_classification_results_returns_stats_report_path_when_requested(self):
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
                    export_stats_report=True,
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
            display.sftp_client = object()

            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmp_dir,
                    remote_hist_dir="/remote/historic",
                    remote_annotated_dir="/remote/annotated",
                    remote_raw_dir="/remote/raw",
                ),
                file_manager=FileManager(),
            )
            raw_target = "118610000000000000001_camera.raw"
            raw_collision = "1186100000000000000012_camera.raw"
            controller.file_manager.sftp_chdir = MagicMock()
            controller.file_manager.sftp_listdir = MagicMock(
                side_effect=[
                    [target_name, survivor_name],
                    [target_name, survivor_name],
                    [raw_target, raw_collision],
                ]
            )
            controller.file_manager.sftp_remove = MagicMock()
            controller.enter_historic_mode = MagicMock()
            controller.exit_historic_mode = MagicMock()
            controller._show_no_images_dialog = MagicMock()

            result = controller.perform_delete_current_piece()

            self.assertTrue(result["ok"])
            self.assertFalse((annotated_dir / target_name).exists())
            self.assertFalse((historic_dir / target_name).exists())
            self.assertTrue((annotated_dir / survivor_name).exists())
            self.assertTrue((historic_dir / survivor_name).exists())
            controller.file_manager.sftp_remove.assert_any_call(
                display.sftp_client,
                f"/remote/raw/{raw_target}",
            )
            removed_paths = [call.args[1] for call in controller.file_manager.sftp_remove.call_args_list]
            self.assertNotIn(f"/remote/raw/{raw_collision}", removed_paths)
            controller.enter_historic_mode.assert_called_once()

    def test_perform_delete_current_piece_without_sftp_preserves_local_and_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            historic_dir = Path(tmp_dir) / "historic"
            historic_dir.mkdir()
            target_name = "12_Cam1_Side1_OK.png"
            (historic_dir / target_name).write_bytes(b"target")
            db = MagicMock()
            display = self._build_display(db=db)
            display.historic_images = [[target_name]]
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )

            result = controller.perform_delete_current_piece()

            self.assertFalse(result["ok"])
            self.assertTrue((historic_dir / target_name).exists())
            db.execute.assert_not_called()
            self.assertIn("SFTP", display.sync_message)

    def test_perform_delete_current_piece_remote_failure_preserves_local_and_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            historic_dir = Path(tmp_dir) / "historic"
            historic_dir.mkdir()
            target_name = "12_Cam1_Side1_OK.png"
            (historic_dir / target_name).write_bytes(b"target")
            db = MagicMock()
            display = self._build_display(db=db)
            display.historic_images = [[target_name]]
            display.sftp_client = object()
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )
            controller.file_manager.sftp_chdir = MagicMock()
            controller.file_manager.sftp_listdir = MagicMock(
                side_effect=[[target_name], [target_name], ["12_camera.raw"]]
            )
            controller.file_manager.sftp_remove = MagicMock(
                side_effect=[None, None, OSError("raw delete failed")]
            )

            result = controller.perform_delete_current_piece()

            self.assertFalse(result["ok"])
            self.assertTrue((historic_dir / target_name).exists())
            db.execute.assert_not_called()
            self.assertIn("raw delete failed", result["error"])

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
                    remote_raw_dir="/remote/raw",
                ),
                file_manager=FileManager(),
            )
            controller.file_manager.sftp_chdir = MagicMock()
            controller.file_manager.sftp_listdir = MagicMock(
                side_effect=[
                    ["remote_historic.png"],
                    ["remote_annotated.png"],
                    ["remote_piece.raw"],
                ]
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
                "/remote/historic/remote_historic.png",
            )
            controller.file_manager.sftp_remove.assert_any_call(
                display.sftp_client,
                "/remote/annotated/remote_annotated.png",
            )
            controller.file_manager.sftp_remove.assert_any_call(
                display.sftp_client,
                "/remote/raw/remote_piece.raw",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

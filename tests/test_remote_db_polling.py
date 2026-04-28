import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from main_controller import ControllerConfig, MainController


class _PollingTestController(MainController):
    def __init__(self, *args, **kwargs):
        self.worker_started_event = threading.Event()
        super().__init__(*args, **kwargs)

    def _remote_db_polling_worker(self):
        self.worker_started_event.set()
        while self.remote_db_stop_event is not None and not self.remote_db_stop_event.is_set():
            time.sleep(0.01)


class TestRemoteDbPolling(unittest.TestCase):
    def _build_display(self):
        display = MagicMock()
        display.db = None
        display.historic_images = []
        display.historic_offset = 0
        display.historic_mode = False
        display.historic_db_registered = False
        display.image_paths = []
        display.exit_requested = False
        display.remote_action_request = None
        display.remote_requested = False
        display.trigger_active = False
        display.file_manager = None
        display.sftp_credentials = None
        display.set_controller = MagicMock()
        display.set_db_connection = MagicMock()
        display.set_db_blocked = MagicMock()
        display.set_sftp_client = MagicMock()
        display.close = MagicMock()
        display.show_image_grid = MagicMock()
        display.download_process = None
        display.download_stop_event = None
        display.annotated_download_process = None
        display.annotated_download_stop_event = None
        return display

    def _build_logger(self):
        logger = MagicMock()
        logger.info = MagicMock()
        logger.warn = MagicMock()
        logger.error = MagicMock()
        return logger

    def _build_db_client(self):
        db = MagicMock()
        db.fetch = MagicMock()
        db.execute = MagicMock()
        db.close = MagicMock()
        return db

    def test_initialize_starts_remote_db_polling(self):
        display = self._build_display()
        logger = self._build_logger()
        controller = MainController(display=display, logger=logger, config=ControllerConfig())
        controller._clear_tmp_display = MagicMock()
        controller.try_connect_db = MagicMock(return_value=False)
        controller.start_remote_db_polling = MagicMock(return_value=True)

        controller.initialize()

        controller.start_remote_db_polling.assert_called_once_with()

    def test_shutdown_stops_remote_db_polling(self):
        display = self._build_display()
        logger = self._build_logger()
        controller = MainController(display=display, logger=logger, config=ControllerConfig())
        controller.stop_remote_db_polling = MagicMock()
        controller.stop_remote_process = MagicMock()
        controller.stop_historic_download_worker = MagicMock()

        controller.shutdown()

        controller.stop_remote_db_polling.assert_called_once_with()
        controller.stop_remote_process.assert_called_once_with("exit")
        controller.stop_historic_download_worker.assert_called_once_with()
        display.close.assert_called_once_with()

    def test_start_remote_db_polling_does_not_duplicate_worker(self):
        display = self._build_display()
        logger = self._build_logger()
        controller = _PollingTestController(
            display=display,
            logger=logger,
            config=ControllerConfig(remote_db_polling_enabled=True),
        )

        self.assertTrue(controller.start_remote_db_polling())
        self.assertTrue(controller.worker_started_event.wait(timeout=1.0))
        first_thread = controller.remote_db_poll_thread

        self.assertFalse(controller.start_remote_db_polling())
        self.assertIs(controller.remote_db_poll_thread, first_thread)

        controller.stop_remote_db_polling()

        self.assertIsNone(controller.remote_db_poll_thread)
        self.assertIsNone(controller.remote_db_stop_event)
        self.assertFalse(first_thread.is_alive())

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_syncs_existing_local_rows_and_deletes_remote(self, remote_db_factory):
        display = self._build_display()
        local_db = self._build_db_client()
        local_db.fetch.return_value = [{"id": 77, "img_name": "img_001.png"}]
        local_db.execute.return_value = 1
        display.db = local_db
        logger = self._build_logger()
        config = ControllerConfig(
            remote_db_table="model_results",
            remote_db_query_limit=25,
            remote_db_target_sync_batch=1,
            remote_db_max_scan_pages=1,
            remote_db_forward_scan_ratio=1.0,
            remote_db_success_interval_sec=0.0,
            remote_db_idle_backoff_sec=2.0,
            remote_db_error_backoff_sec=7.0,
        )
        controller = MainController(
            display=display,
            logger=logger,
            config=config,
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        controller._recalculate_piece_result = MagicMock()
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"id": 101, "img_name": "img_001.png", "class_name": "dent", "confidence": 0.9321}
        ]
        remote_db.execute.return_value = 1
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 1.0)
        remote_db_factory.assert_called_once_with(
            ssh_host="192.168.1.179",
            ssh_port=22,
            ssh_username="vision",
            ssh_password="secret",
        )
        remote_db.fetch.assert_called_once_with(
            'SELECT "id", "img_name", "class_name", "confidence" FROM "model_results" WHERE "id" > %s ORDER BY "id" ASC LIMIT %s',
            (0, 25),
        )
        local_db.fetch.assert_called_once_with(
            "SELECT id, img_name FROM classified_images WHERE img_name = ANY(%s)",
            (["img_001.png"],),
        )
        self.assertEqual(local_db.execute.call_count, 1)
        self.assertEqual(
            local_db.execute.call_args_list[0].args,
            (
                "INSERT INTO classified_image_defects "
                "(classified_image_id, class_name, confidence) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (classified_image_id, class_name, confidence) DO NOTHING",
                (77, "dent", 0.9321),
            ),
        )
        remote_db.execute.assert_called_once_with(
            'DELETE FROM "model_results" WHERE id = ANY(%s)',
            ([101],),
        )
        remote_db.close.assert_not_called()
        self.assertTrue(
            any(
                "Retrieved 1 rows for forward scan using LIMIT 25" in call.args[0]
                for call in logger.debug.call_args_list
            )
        )
        self.assertTrue(
            any("scanned=1, pages=1, candidates=1, matched=1, synced=1, deleted_remote=1" in call.args[0] for call in logger.info.call_args_list)
        )
        self.assertFalse(any('"class_name": "dent"' in call.args[0] for call in logger.info.call_args_list))

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_maps_remote_nok_class_name_to_streaked(self, remote_db_factory):
        display = self._build_display()
        local_db = self._build_db_client()
        local_db.fetch.return_value = [{"id": 77, "img_name": "img_001.png"}]
        local_db.execute.return_value = 1
        display.db = local_db
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_query_limit=25,
                remote_db_target_sync_batch=1,
                remote_db_max_scan_pages=1,
                remote_db_forward_scan_ratio=1.0,
                remote_db_success_interval_sec=0.0,
                remote_db_idle_backoff_sec=2.0,
                remote_db_error_backoff_sec=7.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        controller._recalculate_piece_result = MagicMock()
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"id": 101, "img_name": "img_001.png", "class_name": "NOK", "confidence": 0.9321}
        ]
        remote_db.execute.return_value = 1
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 1.0)
        self.assertEqual(local_db.execute.call_count, 1)
        self.assertEqual(
            local_db.execute.call_args_list[0].args,
            (
                "INSERT INTO classified_image_defects "
                "(classified_image_id, class_name, confidence) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (classified_image_id, class_name, confidence) DO NOTHING",
                (77, "STREAKED", 0.9321),
            ),
        )

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_skips_when_table_is_missing(self, remote_db_factory):
        display = self._build_display()
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(remote_db_table=""),
        )

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, controller.config.remote_db_error_backoff_sec)
        remote_db_factory.assert_not_called()
        self.assertTrue(
            any("configure ControllerConfig.remote_db_table" in call.args[0] for call in logger.warn.call_args_list)
        )

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_keeps_remote_rows_without_local_match(self, remote_db_factory):
        display = self._build_display()
        local_db = self._build_db_client()
        local_db.fetch.return_value = []
        display.db = local_db
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_target_sync_batch=1,
                remote_db_max_scan_pages=1,
                remote_db_forward_scan_ratio=1.0,
                remote_db_idle_backoff_sec=2.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"id": 201, "img_name": "img_missing.png", "class_name": "scratch", "confidence": 0.51}
        ]
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 2.0)
        local_db.execute.assert_not_called()
        remote_db.execute.assert_not_called()
        self.assertTrue(
            any(
                "Missing local classified_images rows during poll: count=1 (examples: img_missing.png)" in call.args[0]
                for call in logger.info.call_args_list
            )
        )
        self.assertTrue(
            any(
                "scanned=1, pages=1" in call.args[0]
                and "matched=0, synced=0, deleted_remote=0" in call.args[0]
                for call in logger.info.call_args_list
            )
        )

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_uses_idle_backoff_and_logs_idle_once_when_empty(self, remote_db_factory):
        display = self._build_display()
        display.db = self._build_db_client()
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_query_limit=25,
                remote_db_target_sync_batch=1,
                remote_db_max_scan_pages=2,
                remote_db_forward_scan_ratio=0.5,
                remote_db_success_interval_sec=0.0,
                remote_db_idle_backoff_sec=3.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        remote_db = MagicMock()
        remote_db.fetch.return_value = []
        remote_db_factory.return_value = remote_db

        first_delay = controller._run_remote_db_poll_iteration()
        second_delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(first_delay, 3.0)
        self.assertEqual(second_delay, 3.0)
        idle_logs = [
            call.args[0]
            for call in logger.info.call_args_list
            if "No remote metadata available to sync" in call.args[0]
        ]
        self.assertEqual(len(idle_logs), 1)
        remote_db_factory.assert_called_once()
        remote_db.close.assert_not_called()

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_reuses_persistent_ssh_tunnel_across_iterations(self, remote_db_factory):
        display = self._build_display()
        display.db = self._build_db_client()
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_query_limit=25,
                remote_db_target_sync_batch=1,
                remote_db_max_scan_pages=1,
                remote_db_forward_scan_ratio=1.0,
                remote_db_success_interval_sec=0.0,
                remote_db_idle_backoff_sec=2.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        remote_db = MagicMock()
        remote_db.fetch.return_value = []
        remote_db_factory.return_value = remote_db

        first_delay = controller._run_remote_db_poll_iteration()
        second_delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(first_delay, 2.0)
        self.assertEqual(second_delay, 2.0)
        self.assertIs(controller.remote_db_client, remote_db)
        remote_db_factory.assert_called_once()
        remote_db.close.assert_not_called()

        controller.stop_remote_db_polling()

        remote_db.close.assert_called_once_with()
        self.assertIsNone(controller.remote_db_client)

    @patch("db.get_remote_db_connection_via_ssh", side_effect=RuntimeError("boom"))
    def test_remote_db_poll_iteration_uses_error_backoff_on_failure(self, remote_db_factory):
        display = self._build_display()
        display.db = self._build_db_client()
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_error_backoff_sec=11.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 11.0)
        remote_db_factory.assert_called_once()
        self.assertTrue(
            any("Poll iteration failed: boom" in call.args[0] for call in logger.error.call_args_list)
        )

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_does_not_delete_remote_rows_when_local_update_fails(self, remote_db_factory):
        display = self._build_display()
        local_db = self._build_db_client()
        local_db.fetch.return_value = [{"id": 77, "img_name": "img_001.png"}]
        local_db.execute.side_effect = RuntimeError("local insert failed")
        display.db = local_db
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_target_sync_batch=1,
                remote_db_max_scan_pages=1,
                remote_db_forward_scan_ratio=1.0,
                remote_db_error_backoff_sec=9.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"id": 101, "img_name": "img_001.png", "class_name": "dent", "confidence": 0.9321}
        ]
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 9.0)
        remote_db.execute.assert_not_called()
        remote_db.close.assert_not_called()
        self.assertTrue(
            any("Poll iteration failed: local insert failed" in call.args[0] for call in logger.error.call_args_list)
        )

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_recreates_tunnel_after_remote_fetch_failure(self, remote_db_factory):
        display = self._build_display()
        display.db = self._build_db_client()
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_error_backoff_sec=11.0,
                remote_db_idle_backoff_sec=2.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        failed_remote_db = MagicMock()
        failed_remote_db.fetch.side_effect = RuntimeError("remote tunnel down")
        healthy_remote_db = MagicMock()
        healthy_remote_db.fetch.return_value = []
        remote_db_factory.side_effect = [failed_remote_db, healthy_remote_db]

        first_delay = controller._run_remote_db_poll_iteration()
        second_delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(first_delay, 11.0)
        self.assertEqual(second_delay, 2.0)
        self.assertEqual(remote_db_factory.call_count, 2)
        failed_remote_db.close.assert_called_once_with()
        healthy_remote_db.close.assert_not_called()
        self.assertIs(controller.remote_db_client, healthy_remote_db)

    @patch("db.get_remote_db_connection_via_ssh")
    def test_remote_db_poll_iteration_uses_forward_cursor_to_reach_later_matches(self, remote_db_factory):
        display = self._build_display()
        local_db = self._build_db_client()
        local_db.fetch.side_effect = [
            [],
            [{"id": 77, "img_name": "img_match.png"}],
        ]
        local_db.execute.return_value = 1
        display.db = local_db
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
                remote_db_query_limit=1,
                remote_db_target_sync_batch=1,
                remote_db_max_scan_pages=2,
                remote_db_forward_scan_ratio=1.0,
            ),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        controller._recalculate_piece_result = MagicMock()
        remote_db = MagicMock()
        remote_db.fetch.side_effect = [
            [{"id": 25, "img_name": "img_missing.png", "class_name": "scratch", "confidence": 0.51}],
            [{"id": 50, "img_name": "img_match.png", "class_name": "dent", "confidence": 0.91}],
        ]
        remote_db.execute.return_value = 1
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 1.0)
        self.assertEqual(
            remote_db.fetch.call_args_list[0].args,
            (
                'SELECT "id", "img_name", "class_name", "confidence" FROM "model_results" WHERE "id" > %s ORDER BY "id" ASC LIMIT %s',
                (0, 1),
            ),
        )
        self.assertEqual(
            remote_db.fetch.call_args_list[1].args,
            (
                'SELECT "id", "img_name", "class_name", "confidence" FROM "model_results" WHERE "id" > %s ORDER BY "id" ASC LIMIT %s',
                (25, 1),
            ),
        )
        remote_db.execute.assert_called_once_with(
            'DELETE FROM "model_results" WHERE id = ANY(%s)',
            ([50],),
        )
        self.assertEqual(controller.remote_db_forward_cursor_id, 50)

    def test_upsert_classification_keeps_created_at_as_db_default(self):
        display = self._build_display()
        logger = self._build_logger()
        db = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = {"id": 123}
        db_context = MagicMock()
        db_context.__enter__.return_value = cursor
        db_context.__exit__.return_value = None
        db.get_cursor.return_value = db_context

        controller = MainController(display=display, logger=logger, config=ControllerConfig())

        controller._upsert_classification("118610000000000000001_Cam1_Side1_OK.png", "OK", db_client=db)

        classified_query = cursor.execute.call_args_list[1].args[0]
        self.assertIn("(img_name, operator_result, model_result, piece_id)", classified_query)
        self.assertNotIn("created_at", classified_query.lower())
        self.assertNotIn("class_name", classified_query.lower())
        self.assertNotIn("confidence", classified_query.lower())

    def test_sync_remote_rows_recalculates_piece_result_for_synced_jsn(self):
        display = self._build_display()
        display.db = self._build_db_client()
        logger = self._build_logger()
        controller = MainController(display=display, logger=logger, config=ControllerConfig())
        local_db = self._build_db_client()
        remote_db = MagicMock()
        local_db.fetch.return_value = [{"id": 77, "img_name": "11861001_Cam1_Side1_OK.png"}]
        controller._recalculate_piece_result = MagicMock()

        summary = controller._sync_remote_rows_into_local_classified_images(
            matched_rows=[
                {
                    "remote_id": 500,
                    "classified_image_id": 77,
                    "img_name": "11861001_Cam1_Side1_OK.png",
                    "class_name": "dent",
                    "confidence": 0.9321,
                }
            ],
            missing_local=[],
            local_db=local_db,
            remote_db=remote_db,
        )

        self.assertEqual(summary["synced_count"], 1)
        controller._recalculate_piece_result.assert_called_once_with(
            "11861001",
            db_client=local_db,
        )

    def test_recalculate_piece_result_refreshes_piece_result_defects_with_single_best_defect(self):
        display = self._build_display()
        logger = self._build_logger()
        db = self._build_db_client()
        db.fetch.return_value = [{"id": 321}]
        controller = MainController(display=display, logger=logger, config=ControllerConfig())

        controller._recalculate_piece_result("11861001", db_client=db)

        self.assertEqual(db.execute.call_count, 3)
        update_query = db.execute.call_args_list[0].args[0]
        delete_query = db.execute.call_args_list[1].args[0]
        insert_query = db.execute.call_args_list[2].args[0]

        self.assertIn("operator_result = COALESCE(", update_query)
        self.assertIn("model_result = COALESCE(", update_query)
        self.assertIn("DELETE FROM piece_result_defects WHERE piece_result_id = %s", delete_query)
        self.assertIn(
            "INSERT INTO piece_result_defects (piece_result_id, class_name, confidence)",
            insert_query,
        )
        self.assertIn("FROM (", insert_query)
        self.assertIn("FROM classified_image_defects cid", insert_query)
        self.assertIn("JOIN classified_images ci ON ci.id = cid.classified_image_id", insert_query)
        self.assertIn("UPPER(cid.class_name) <> 'OK'", insert_query)
        self.assertIn("NOT EXISTS (", insert_query)
        self.assertIn("ci_non_ok.piece_id = %s", insert_query)
        self.assertIn("ORDER BY cid.confidence DESC, cid.created_at DESC, cid.id DESC", insert_query)
        self.assertIn("LIMIT 1", insert_query)
        self.assertEqual(db.execute.call_args_list[2].args[1], (321, 321, 321))


if __name__ == "__main__":
    unittest.main(verbosity=2)

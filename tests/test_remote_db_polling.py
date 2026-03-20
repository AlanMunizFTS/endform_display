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
            config=ControllerConfig(),
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
        local_db.fetch.return_value = [{"img_name": "img_001.png"}]
        local_db.execute.return_value = 1
        display.db = local_db
        logger = self._build_logger()
        config = ControllerConfig(
            remote_db_table="model_results",
            remote_db_query_limit=25,
            remote_db_success_interval_sec=0.0,
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
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"img_name": "img_001.png", "class_name": "dent", "confidence": 0.9321}
        ]
        remote_db.execute.return_value = 1
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 0.0)
        remote_db_factory.assert_called_once_with(
            ssh_host="192.168.1.179",
            ssh_port=22,
            ssh_username="vision",
            ssh_password="secret",
        )
        remote_db.fetch.assert_called_once_with(
            'SELECT "img_name", "class_name", "confidence" FROM "model_results" ORDER BY "created_at" ASC LIMIT 25'
        )
        local_db.fetch.assert_called_once_with(
            "SELECT img_name FROM classified_images WHERE img_name = ANY(%s)",
            (["img_001.png"],),
        )
        local_db.execute.assert_called_once_with(
            "UPDATE classified_images SET class_name = %s, confidence = %s WHERE img_name = %s",
            ("dent", 0.9321, "img_001.png"),
        )
        remote_db.execute.assert_called_once_with(
            'DELETE FROM "model_results" WHERE img_name = ANY(%s)',
            (["img_001.png"],),
        )
        remote_db.close.assert_called_once_with()
        self.assertTrue(
            any(
                'Executing query: SELECT "img_name", "class_name", "confidence" FROM "model_results" ORDER BY "created_at" ASC LIMIT 25'
                in call.args[0]
                for call in logger.info.call_args_list
            )
        )
        self.assertTrue(
            any("Retrieved 1 rows using LIMIT 25" in call.args[0] for call in logger.info.call_args_list)
        )
        self.assertTrue(
            any("candidates=1, matched=1, updated=1, deleted_remote=1" in call.args[0] for call in logger.info.call_args_list)
        )
        self.assertTrue(
            any('"class_name": "dent"' in call.args[0] for call in logger.info.call_args_list)
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
            config=ControllerConfig(remote_db_table="model_results"),
            sftp_credentials={
                "hostname": "192.168.1.179",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        )
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"img_name": "img_missing.png", "class_name": "scratch", "confidence": 0.51}
        ]
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 0.0)
        local_db.execute.assert_not_called()
        remote_db.execute.assert_not_called()
        self.assertTrue(
            any("img_missing.png" in call.args[0] for call in logger.info.call_args_list)
        )
        self.assertTrue(
            any("candidates=1, matched=0, updated=0, deleted_remote=0" in call.args[0] for call in logger.info.call_args_list)
        )

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
        local_db.fetch.return_value = [{"img_name": "img_001.png"}]
        local_db.execute.side_effect = RuntimeError("local update failed")
        display.db = local_db
        logger = self._build_logger()
        controller = MainController(
            display=display,
            logger=logger,
            config=ControllerConfig(
                remote_db_table="model_results",
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
            {"img_name": "img_001.png", "class_name": "dent", "confidence": 0.9321}
        ]
        remote_db_factory.return_value = remote_db

        delay = controller._run_remote_db_poll_iteration()

        self.assertEqual(delay, 9.0)
        remote_db.execute.assert_not_called()
        self.assertTrue(
            any("Poll iteration failed: local update failed" in call.args[0] for call in logger.error.call_args_list)
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

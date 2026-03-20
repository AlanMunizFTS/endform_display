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
    def test_remote_db_poll_iteration_logs_rows_and_closes_connection(self, remote_db_factory):
        display = self._build_display()
        logger = self._build_logger()
        config = ControllerConfig(
            remote_db_table="model_results",
            remote_db_query_limit=10,
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
        remote_db.fetch.return_value = [{"id": 1, "status": "OK"}]
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
            'SELECT "img_name", "class_name", "confidence" FROM "model_results" LIMIT 10'
        )
        remote_db.close.assert_called_once_with()
        self.assertTrue(
            any(
                'Executing query: SELECT "img_name", "class_name", "confidence" FROM "model_results" LIMIT 10'
                in call.args[0]
                for call in logger.info.call_args_list
            )
        )
        self.assertTrue(
            any("Retrieved 1 rows using LIMIT 10" in call.args[0] for call in logger.info.call_args_list)
        )
        self.assertTrue(
            any('"status": "OK"' in call.args[0] for call in logger.info.call_args_list)
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

    @patch("db.get_remote_db_connection_via_ssh", side_effect=RuntimeError("boom"))
    def test_remote_db_poll_iteration_uses_error_backoff_on_failure(self, remote_db_factory):
        display = self._build_display()
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

import sys
import tempfile
import types
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, patch

from file_manager import FileManager
from main_controller import (
    ControllerConfig,
    MainController,
    _download_images_background_worker,
)


class _DisplayNumericJsnStub:
    def __init__(self):
        self.db = MagicMock()
        self.historic_images = []
        self.historic_offset = 0
        self.historic_mode = False
        self.historic_db_registered = False
        self.historic_index_rescan_interval = 1.0
        self.temp_results = {}
        self.available_jsns = []
        self.filtered_suggestions = []
        self.search_jsn = ""
        self.search_active = False
        self.selected_suggestion_idx = -1
        self.show_reset_confirm = False
        self.show_delete_confirm = False
        self.show_rebuild_confirm = False
        self.show_piece_date_dialog = False
        self._db_registered_images = set()
        self._db_result_cache = {}
        self._image_cache = {}
        self._historic_index_cache = None
        self._historic_jsn_cache = []
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self.sftp_client = None
        self.set_db_connection = MagicMock()


class TestMainControllerNumericJsn(unittest.TestCase):
    def test_list_local_image_names_requires_numeric_jsn_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "annotated"
            image_dir.mkdir()
            for name in (
                "22700001_side_OK.png",
                "1004_side_cam1.png",
                "11861-0007_side_OK.png",
                "JSN001_side_OK.png",
                "notes.txt",
            ):
                (image_dir / name).write_bytes(b"x")

            controller = MainController(
                display=_DisplayNumericJsnStub(),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )

            result = controller._list_local_image_names(
                str(image_dir),
                require_jsn_prefix=True,
            )

            self.assertEqual(
                sorted(result),
                ["1004_side_cam1.png", "22700001_side_OK.png"],
            )

    def test_load_historic_index_ignores_non_numeric_jsn_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotated_dir = Path(tmp_dir) / "annotated"
            historic_dir = Path(tmp_dir) / "historic"
            annotated_dir.mkdir()
            historic_dir.mkdir()

            for name in (
                "22700001_side_OK.png",
                "1004_side_cam1.png",
                "11861-0007_side_OK.png",
                "JSN001_side_OK.png",
            ):
                (annotated_dir / name).write_bytes(b"x")

            controller = MainController(
                display=_DisplayNumericJsnStub(),
                config=ControllerConfig(temp_dir=tmp_dir),
                file_manager=FileManager(),
            )

            historic_index = controller._load_historic_index(force_rescan=True)

            self.assertEqual(
                historic_index,
                [["22700001_side_OK.png"], ["1004_side_cam1.png"]],
            )

    @patch("db.get_remote_db_connection_via_ssh")
    def test_download_images_background_worker_downloads_only_jsns_ready_in_remote_db(
        self, remote_db_factory
    ):
        stop_event = Event()
        fake_logger = MagicMock()
        fake_sftp_client = MagicMock()
        fake_ssh_client = MagicMock()
        fake_ssh_client.open_sftp.return_value = fake_sftp_client
        remote_db = MagicMock()
        remote_db.fetch.return_value = [
            {"jsn": "22700003"},
            {"jsn": "22700001"},
        ]
        remote_db_factory.return_value = remote_db

        fake_manager = MagicMock()
        fake_manager.exists.return_value = True
        fake_manager.listdir.return_value = []
        fake_manager.sftp_listdir.return_value = [
            "22700003_front_OK.png",
            "22700003_side_OK.png",
            "22700002_side_OK.png",
            "22700001_side_OK.png",
            "11861-0007_side_OK.png",
            "JSN001_side_OK.png",
            "readme.txt",
        ]
        fake_manager.join.side_effect = lambda directory, name: str(Path(directory) / name)

        download_counter = {"count": 0}

        def _fake_get(_client, _remote_name, _local_name):
            download_counter["count"] += 1
            if download_counter["count"] >= 3:
                stop_event.set()

        fake_manager.sftp_get.side_effect = _fake_get

        fake_paramiko = types.SimpleNamespace(
            SSHClient=MagicMock(return_value=fake_ssh_client),
            AutoAddPolicy=MagicMock(return_value=object()),
        )

        with patch("main_controller.install_print_logger"), patch(
            "main_controller.get_logger",
            return_value=fake_logger,
        ), patch(
            "main_controller.FileManager",
            return_value=fake_manager,
        ), patch.dict(
            sys.modules,
            {"paramiko": fake_paramiko},
        ):
            _download_images_background_worker(
                hostname="host",
                port=22,
                username="user",
                password="pwd",
                remote_dir="/remote",
                local_temp_dir="C:\\tmp\\images",
                check_interval=0,
                reconnect_interval=0,
                stop_event=stop_event,
                worker_label="TEST_SYNC",
                validate_remote_jsn=True,
            )

        self.assertEqual(
            [call.args[1] for call in fake_manager.sftp_get.call_args_list],
            [
                "22700003_side_OK.png",
                "22700003_front_OK.png",
                "22700001_side_OK.png",
            ],
        )
        remote_db_factory.assert_called_once_with(
            ssh_host="host",
            ssh_port=22,
            ssh_username="user",
            ssh_password="pwd",
        )
        remote_db.fetch.assert_called_once_with(
            'SELECT CAST("jsn" AS TEXT) AS jsn FROM "pieces_out" WHERE CAST("jsn" AS TEXT) = ANY(%s) AND "status" = %s',
            (["22700003", "22700002", "22700001"], 1),
        )
        remote_db.close.assert_called_once_with()

    @patch("db.get_remote_db_connection_via_ssh")
    def test_download_images_background_worker_downloads_all_candidates_when_remote_validation_disabled(
        self, remote_db_factory
    ):
        stop_event = Event()
        fake_logger = MagicMock()
        fake_sftp_client = MagicMock()
        fake_ssh_client = MagicMock()
        fake_ssh_client.open_sftp.return_value = fake_sftp_client

        fake_manager = MagicMock()
        fake_manager.exists.return_value = True
        fake_manager.listdir.return_value = []
        fake_manager.sftp_listdir.return_value = [
            "22700003_front_OK.png",
            "22700003_side_OK.png",
            "22700002_side_OK.png",
            "22700001_side_OK.png",
            "11861-0007_side_OK.png",
            "JSN001_side_OK.png",
            "readme.txt",
        ]
        fake_manager.join.side_effect = lambda directory, name: str(Path(directory) / name)

        download_counter = {"count": 0}

        def _fake_get(_client, _remote_name, _local_name):
            download_counter["count"] += 1
            if download_counter["count"] >= 4:
                stop_event.set()

        fake_manager.sftp_get.side_effect = _fake_get

        fake_paramiko = types.SimpleNamespace(
            SSHClient=MagicMock(return_value=fake_ssh_client),
            AutoAddPolicy=MagicMock(return_value=object()),
        )

        with patch("main_controller.install_print_logger"), patch(
            "main_controller.get_logger",
            return_value=fake_logger,
        ), patch(
            "main_controller.FileManager",
            return_value=fake_manager,
        ), patch.dict(
            sys.modules,
            {"paramiko": fake_paramiko},
        ):
            _download_images_background_worker(
                hostname="host",
                port=22,
                username="user",
                password="pwd",
                remote_dir="/remote",
                local_temp_dir="C:\\tmp\\images",
                check_interval=0,
                reconnect_interval=0,
                stop_event=stop_event,
                worker_label="TEST_SYNC",
                validate_remote_jsn=False,
            )

        self.assertEqual(
            [call.args[1] for call in fake_manager.sftp_get.call_args_list],
            [
                "22700003_side_OK.png",
                "22700003_front_OK.png",
                "22700002_side_OK.png",
                "22700001_side_OK.png",
            ],
        )
        remote_db_factory.assert_not_called()

    @patch("db.get_remote_db_connection_via_ssh", side_effect=RuntimeError("db down"))
    def test_download_images_background_worker_skips_download_when_remote_db_fails(
        self, _remote_db_factory
    ):
        stop_event = Event()
        fake_logger = MagicMock()
        fake_logger.error.side_effect = lambda *args, **kwargs: stop_event.set()
        fake_sftp_client = MagicMock()
        fake_ssh_client = MagicMock()
        fake_ssh_client.open_sftp.return_value = fake_sftp_client

        fake_manager = MagicMock()
        fake_manager.exists.return_value = True
        fake_manager.listdir.return_value = []
        fake_manager.sftp_listdir.return_value = [
            "22700003_side_OK.png",
            "22700002_side_OK.png",
            "readme.txt",
        ]
        fake_manager.join.side_effect = lambda directory, name: str(Path(directory) / name)

        fake_paramiko = types.SimpleNamespace(
            SSHClient=MagicMock(return_value=fake_ssh_client),
            AutoAddPolicy=MagicMock(return_value=object()),
        )

        with patch("main_controller.install_print_logger"), patch(
            "main_controller.get_logger",
            return_value=fake_logger,
        ), patch(
            "main_controller.FileManager",
            return_value=fake_manager,
        ), patch.dict(
            sys.modules,
            {"paramiko": fake_paramiko},
        ):
            _download_images_background_worker(
                hostname="host",
                port=22,
                username="user",
                password="pwd",
                remote_dir="/remote",
                local_temp_dir="C:\\tmp\\images",
                check_interval=0,
                reconnect_interval=0,
                stop_event=stop_event,
                worker_label="TEST_SYNC",
                validate_remote_jsn=True,
            )

        fake_manager.sftp_get.assert_not_called()
        self.assertTrue(
            any("Remote DB validation failed" in str(call.args[0]) for call in fake_logger.error.call_args_list)
        )

    def test_download_images_background_worker_logs_idle_summary_only_once_for_unchanged_state(self):
        stop_event = Event()
        fake_logger = MagicMock()
        fake_sftp_client = MagicMock()
        fake_ssh_client = MagicMock()
        fake_ssh_client.open_sftp.return_value = fake_sftp_client

        fake_manager = MagicMock()
        fake_manager.exists.return_value = True
        fake_manager.listdir.return_value = ["22700003_side_OK.png"]
        fake_manager.join.side_effect = lambda directory, name: str(Path(directory) / name)

        poll_counter = {"count": 0}

        def _fake_sftp_listdir(_client):
            poll_counter["count"] += 1
            if poll_counter["count"] >= 2:
                stop_event.set()
            return ["22700003_side_OK.png"]

        fake_manager.sftp_listdir.side_effect = _fake_sftp_listdir

        fake_paramiko = types.SimpleNamespace(
            SSHClient=MagicMock(return_value=fake_ssh_client),
            AutoAddPolicy=MagicMock(return_value=object()),
        )

        with patch("main_controller.install_print_logger"), patch(
            "main_controller.get_logger",
            return_value=fake_logger,
        ), patch(
            "main_controller.FileManager",
            return_value=fake_manager,
        ), patch.dict(
            sys.modules,
            {"paramiko": fake_paramiko},
        ):
            _download_images_background_worker(
                hostname="host",
                port=22,
                username="user",
                password="pwd",
                remote_dir="/remote",
                local_temp_dir="C:\\tmp\\images",
                check_interval=0,
                reconnect_interval=0,
                stop_event=stop_event,
                worker_label="TEST_SYNC",
                validate_remote_jsn=False,
            )

        summary_logs = [
            call.args[0]
            for call in fake_logger.info.call_args_list
            if "Sync summary:" in str(call.args[0])
        ]
        self.assertEqual(len(summary_logs), 1)
        self.assertIn("candidate_jsns=0", summary_logs[0])
        self.assertIn("downloaded=0", summary_logs[0])

    def test_download_images_background_worker_logs_summary_again_when_sync_state_changes(self):
        stop_event = Event()
        fake_logger = MagicMock()
        fake_sftp_client = MagicMock()
        fake_ssh_client = MagicMock()
        fake_ssh_client.open_sftp.return_value = fake_sftp_client

        fake_manager = MagicMock()
        fake_manager.exists.return_value = True
        fake_manager.listdir.return_value = ["22700003_side_OK.png"]
        fake_manager.join.side_effect = lambda directory, name: str(Path(directory) / name)

        remote_file_batches = [
            ["22700003_side_OK.png"],
            ["22700003_side_OK.png", "22700004_side_OK.png"],
        ]
        poll_counter = {"count": 0}

        def _fake_sftp_listdir(_client):
            batch_idx = min(poll_counter["count"], len(remote_file_batches) - 1)
            poll_counter["count"] += 1
            return remote_file_batches[batch_idx]

        def _fake_get(_client, _remote_name, _local_name):
            stop_event.set()

        fake_manager.sftp_listdir.side_effect = _fake_sftp_listdir
        fake_manager.sftp_get.side_effect = _fake_get

        fake_paramiko = types.SimpleNamespace(
            SSHClient=MagicMock(return_value=fake_ssh_client),
            AutoAddPolicy=MagicMock(return_value=object()),
        )

        with patch("main_controller.install_print_logger"), patch(
            "main_controller.get_logger",
            return_value=fake_logger,
        ), patch(
            "main_controller.FileManager",
            return_value=fake_manager,
        ), patch.dict(
            sys.modules,
            {"paramiko": fake_paramiko},
        ):
            _download_images_background_worker(
                hostname="host",
                port=22,
                username="user",
                password="pwd",
                remote_dir="/remote",
                local_temp_dir="C:\\tmp\\images",
                check_interval=0,
                reconnect_interval=0,
                stop_event=stop_event,
                worker_label="TEST_SYNC",
                validate_remote_jsn=False,
            )

        summary_logs = [
            call.args[0]
            for call in fake_logger.info.call_args_list
            if "Sync summary:" in str(call.args[0])
        ]
        self.assertEqual(len(summary_logs), 2)
        self.assertIn("downloaded=0", summary_logs[0])
        self.assertIn("downloaded=1", summary_logs[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

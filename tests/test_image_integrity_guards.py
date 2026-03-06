import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from display_window import DisplayWindow
from file_manager import FileManager
from main_controller import (
    ControllerConfig,
    MainController,
    _download_live_images_local_impl,
    _download_live_images_remote_impl,
    _sftp_get_with_cleanup_retry,
)
from paths_config import HISTORIC_SUBDIR_NAME


class _DisplayStub:
    def __init__(self):
        self.db = None
        self._historic_index_cache = None
        self._historic_jsn_cache = []
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self.historic_index_rescan_interval = 0.0
        self.historic_db_registered = False


class TestImageIntegrityGuards(unittest.TestCase):
    def setUp(self):
        self.fm = FileManager()

    def test_sftp_get_retry_cleans_partial_and_retries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = self.fm.join(tmpdir, "img.png")
            sftp_client = MagicMock()
            call_count = {"value": 0}

            def _fake_get(_remote, local):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    with open(local, "wb"):
                        pass
                    return

                self.assertFalse(os.path.exists(local))
                with open(local, "wb") as f:
                    f.write(b"ok")

            sftp_client.get.side_effect = _fake_get

            _sftp_get_with_cleanup_retry(
                file_manager=self.fm,
                sftp_client=sftp_client,
                remote_path="img.png",
                local_path=local_path,
                max_attempts=2,
            )

            self.assertEqual(sftp_client.get.call_count, 2)
            self.assertEqual(self.fm.getsize(local_path), 2)

    def test_live_local_scan_ignores_zero_byte_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_name = "118610001_side_cam1.png"
            empty_name = "118610002_side_cam1.png"
            valid_path = self.fm.join(tmpdir, valid_name)
            empty_path = self.fm.join(tmpdir, empty_name)

            with open(valid_path, "wb") as f:
                f.write(b"valid")
            with open(empty_path, "wb"):
                pass

            rotation_state = {"current_offset": 0, "cached_images": None}
            result = _download_live_images_local_impl(
                file_manager=self.fm,
                local_path=tmpdir,
                rotation_state=rotation_state,
                logger=MagicMock(),
                image_extensions=(".png", ".jpg", ".jpeg", ".bmp"),
                live_rescan_interval_sec=0.0,
                live_batch_rotation_interval_sec=1.0,
                max_images=7,
            )

            self.assertEqual(result, [valid_path])

    def test_historic_index_excludes_zero_byte_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            historic_dir = self.fm.join(tmpdir, HISTORIC_SUBDIR_NAME)
            self.fm.makedirs(historic_dir, exist_ok=True)

            valid_name = "118610123_side_cam1.png"
            empty_name = "118610123_front_cam2.png"
            valid_path = self.fm.join(historic_dir, valid_name)
            empty_path = self.fm.join(historic_dir, empty_name)

            with open(valid_path, "wb") as f:
                f.write(b"valid")
            with open(empty_path, "wb"):
                pass

            display = _DisplayStub()
            controller = MainController(
                display=display,
                logger=MagicMock(),
                config=ControllerConfig(temp_dir=tmpdir),
                file_manager=self.fm,
            )

            historic_groups = controller._load_historic_index(force_rescan=True)
            self.assertEqual(historic_groups, [[valid_name]])

    @patch("display_window.get_logger")
    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_failed_decode_cache_skips_redecode_when_mtime_unchanged(
        self, _db_mock, get_logger_mock
    ):
        logger_mock = MagicMock()
        get_logger_mock.return_value = logger_mock
        file_manager = MagicMock()
        file_manager.getmtime.side_effect = [100.0, 100.0, 101.0]
        file_manager.getsize.return_value = 128
        file_manager.read_image.return_value = None

        window = DisplayWindow(file_manager=file_manager, sftp_client=None)

        self.assertIsNone(window._get_cached_image("img.png"))
        self.assertIsNone(window._get_cached_image("img.png"))
        self.assertIsNone(window._get_cached_image("img.png"))
        self.assertEqual(file_manager.read_image.call_count, 2)
        self.assertEqual(logger_mock.warn.call_count, 2)
        self.assertIn("decode_error_count=1", logger_mock.warn.call_args_list[0].args[0])
        self.assertIn("decode_error_count=2", logger_mock.warn.call_args_list[1].args[0])

    @patch("display_window.get_logger")
    @patch("display_window.get_db_connection", return_value=MagicMock())
    def test_zero_byte_image_skips_decode(self, _db_mock, get_logger_mock):
        logger_mock = MagicMock()
        get_logger_mock.return_value = logger_mock
        file_manager = MagicMock()
        file_manager.getmtime.return_value = 200.0
        file_manager.getsize.return_value = 0
        file_manager.read_image.return_value = None

        window = DisplayWindow(file_manager=file_manager, sftp_client=None)

        self.assertIsNone(window._get_cached_image("zero.png"))
        self.assertIsNone(window._get_cached_image("zero.png"))
        file_manager.read_image.assert_not_called()
        logger_mock.warn.assert_called_once()
        self.assertIn("zero_byte_count=1", logger_mock.warn.call_args.args[0])

    def test_live_remote_transfer_error_logs_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sftp_client = MagicMock()
            sftp_client.get.side_effect = RuntimeError("network issue")
            logger = MagicMock()
            file_manager = FileManager()

            class _App:
                def __init__(self):
                    self.sftp_client = sftp_client
                    self.file_manager = file_manager

                def list_remote_files(self, _remote_dir):
                    return ["118610001_side_cam1.png"]

                def ensure_remote_dir(self, _remote_dir):
                    return None

                def join_remote_path(self, remote_dir, filename):
                    return f"{remote_dir}/{filename}"

                def upload_file(self, _local, _remote):
                    return None

            rotation_state = {"current_offset": 0}
            downloaded = _download_live_images_remote_impl(
                app=_App(),
                remote_path="/remote/live",
                local_path=tmpdir,
                remote_hist_dir="/remote/hist",
                rotation_state=rotation_state,
                logger=logger,
                image_extensions=(".png",),
                max_images=7,
            )

            self.assertEqual(downloaded, [])
            self.assertEqual(rotation_state.get("transfer_error_count"), 1)
            logger.warn.assert_called_once()
            self.assertIn("transfer_error_count=1", logger.warn.call_args.args[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

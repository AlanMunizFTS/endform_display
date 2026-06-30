import os
import tempfile
import unittest
from unittest.mock import MagicMock

from main_controller import ControllerConfig, MainController
from paths_config import ANNOTATED_SUBDIR_NAME, HISTORIC_SUBDIR_NAME


class _DisplayStub:
    def __init__(self, db, sftp_client):
        self.db = db
        self.sftp_client = sftp_client
        self.sftp_credentials = None

        self.historic_mode = True
        self.historic_offset = 2
        self.historic_images = [["sample.png"]]
        self.search_jsn = "11861"
        self.search_active = True
        self.filtered_suggestions = ["11861"]
        self.selected_suggestion_idx = 0
        self.show_reset_confirm = True
        self.show_delete_confirm = True
        self.show_rebuild_confirm = True
        self.show_piece_date_dialog = True

        self.temp_results = {"sample.png": "NOK"}
        self.available_jsns = ["11861"]
        self.historic_db_registered = True
        self._db_registered_images = {"sample.png"}
        self._historic_index_cache = {"cached": True}
        self._historic_index_mtime = 1.0
        self._historic_index_last_scan = 1.0
        self._historic_jsn_cache = ["11861"]
        self._db_result_cache = {"sample.png": "NOK"}
        self._image_cache = {"sample.png": object()}

        self.download_process = None
        self.download_stop_event = None
        self.annotated_download_process = None
        self.annotated_download_stop_event = None


class TestMainControllerReset(unittest.TestCase):
    def test_perform_reset_clears_local_and_remote_annotated_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            historic_dir = os.path.join(tmpdir, HISTORIC_SUBDIR_NAME)
            annotated_dir = os.path.join(tmpdir, ANNOTATED_SUBDIR_NAME)
            os.makedirs(historic_dir, exist_ok=True)
            os.makedirs(annotated_dir, exist_ok=True)

            with open(os.path.join(historic_dir, "historic_a.png"), "w", encoding="utf-8") as f:
                f.write("historic")
            with open(os.path.join(annotated_dir, "annotated_a.png"), "w", encoding="utf-8") as f:
                f.write("annotated")

            db = MagicMock()
            db.execute.return_value = 2

            sftp_client = MagicMock()
            sftp_client.listdir.side_effect = [
                ["remote_hist_a.png"],
                ["remote_annotated_a.png"],
            ]

            display = _DisplayStub(db=db, sftp_client=sftp_client)
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmpdir,
                    historic_download_check_interval=17,
                ),
            )
            controller.stop_historic_download_worker = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()

            result = controller.perform_reset()

            self.assertEqual(result, {"ok": True})
            self.assertTrue(os.path.isdir(historic_dir))
            self.assertTrue(os.path.isdir(annotated_dir))
            self.assertEqual(os.listdir(historic_dir), [])
            self.assertEqual(os.listdir(annotated_dir), [])

            self.assertEqual(db.execute.call_args_list[0].args, ("DELETE FROM img_results",))
            self.assertEqual(db.execute.call_args_list[1].args, ("DELETE FROM model_results",))
            self.assertEqual(
                db.execute.call_args_list[2].args,
                ("DELETE FROM remote_model_results_pending",),
            )
            self.assertEqual(
                db.execute.call_args_list[3].args,
                ("DELETE FROM remote_sync_state",),
            )
            sftp_client.chdir.assert_any_call(controller.config.remote_hist_dir)
            sftp_client.chdir.assert_any_call(controller.config.remote_annotated_dir)
            sftp_client.remove.assert_any_call(
                f"{controller.config.remote_hist_dir}/remote_hist_a.png"
            )
            sftp_client.remove.assert_any_call(
                f"{controller.config.remote_annotated_dir}/remote_annotated_a.png"
            )
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmpdir,
                check_interval=17,
            )

    def test_perform_reset_sets_sftp_timeout_before_remote_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, HISTORIC_SUBDIR_NAME), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, ANNOTATED_SUBDIR_NAME), exist_ok=True)

            db = MagicMock()
            db.execute.return_value = 0

            sftp_channel = MagicMock()
            sftp_client = MagicMock()
            sftp_client.get_channel.return_value = sftp_channel
            sftp_client.listdir.side_effect = [[], []]

            display = _DisplayStub(db=db, sftp_client=sftp_client)
            controller = MainController(
                display=display,
                config=ControllerConfig(
                    temp_dir=tmpdir,
                    reset_sftp_operation_timeout_sec=12.5,
                ),
            )
            controller.stop_historic_download_worker = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()

            result = controller.perform_reset()

            self.assertEqual(result, {"ok": True})
            sftp_channel.settimeout.assert_any_call(12.5)

    def test_perform_reset_reports_exception_type_when_remote_error_message_is_blank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, HISTORIC_SUBDIR_NAME), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, ANNOTATED_SUBDIR_NAME), exist_ok=True)

            db = MagicMock()
            db.execute.return_value = 0

            sftp_client = MagicMock()
            sftp_client.listdir.side_effect = TimeoutError()

            display = _DisplayStub(db=db, sftp_client=sftp_client)
            controller = MainController(
                display=display,
                config=ControllerConfig(temp_dir=tmpdir),
            )
            controller.stop_historic_download_worker = MagicMock()
            controller.start_historic_download_on_startup = MagicMock()

            result = controller.perform_reset()

            self.assertFalse(result["ok"])
            self.assertIn("TimeoutError", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

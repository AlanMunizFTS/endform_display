import os
import tempfile
import unittest
from unittest.mock import MagicMock

from main_controller import ControllerConfig, MainController
from paths_config import HISTORIC_SUBDIR_NAME


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


class TestMainControllerReset(unittest.TestCase):
    def test_perform_reset_clears_local_and_remote_historic_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            historic_dir = os.path.join(tmpdir, HISTORIC_SUBDIR_NAME)
            os.makedirs(historic_dir, exist_ok=True)

            with open(os.path.join(historic_dir, "historic_a.png"), "w", encoding="utf-8") as f:
                f.write("historic")

            db = MagicMock()
            db.execute.return_value = 2

            sftp_client = MagicMock()
            sftp_client.listdir.side_effect = [
                ["remote_hist_a.png"],
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
            self.assertEqual(os.listdir(historic_dir), [])

            db.execute.assert_called_once_with("DELETE FROM img_results")
            sftp_client.chdir.assert_any_call(controller.config.remote_hist_dir)
            sftp_client.remove.assert_any_call(
                f"{controller.config.remote_hist_dir}/remote_hist_a.png"
            )
            controller.stop_historic_download_worker.assert_called_once_with()
            controller.start_historic_download_on_startup.assert_called_once_with(
                tmpdir,
                check_interval=17,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

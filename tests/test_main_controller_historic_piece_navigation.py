import unittest
from unittest.mock import MagicMock

from main_controller import MainController


class _DisplayPieceStub:
    def __init__(self):
        self.db = MagicMock()
        self.sftp_client = None
        self.sftp_credentials = None
        self.historic_mode = True
        self.historic_images = [
            ["1004_side_cam1.png"],
            ["1003_side_cam1.png"],
            ["1002_side_cam1.png"],
            ["1001_side_cam1.png"],
        ]
        self.historic_offset = 1
        self.search_active = False
        self.filtered_suggestions = []
        self.selected_suggestion_idx = -1
        self.show_piece_date_dialog = False
        self.show_piece_number_dialog = False
        self.show_reset_confirm = False
        self.show_delete_confirm = False
        self.show_rebuild_confirm = False
        self.show_no_images_dialog = False
        self.no_images_dialog_message = ""
        self.piece_number_dialog_input = ""
        self.piece_number_dialog_replace_on_input = False
        self.temp_results = {}
        self.available_jsns = []
        self._db_result_cache = {}
        self._db_registered_images = set()
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self._historic_jsn_cache = []
        self._image_cache = {}


class TestMainControllerHistoricPieceNavigation(unittest.TestCase):
    def test_go_to_historic_piece_number_uses_displayed_piece_numbering(self):
        display = _DisplayPieceStub()
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(return_value=list(display.historic_images))
        controller._refresh_historic_index_async = MagicMock()

        moved = controller.go_to_historic_piece_number(1, show_missing_dialog=True)

        self.assertTrue(moved)
        self.assertEqual(display.historic_offset, 3)
        self.assertTrue(display.historic_mode)
        controller._refresh_historic_index_async.assert_called_once_with()

    def test_open_piece_number_dialog_prefills_current_piece_number(self):
        display = _DisplayPieceStub()
        controller = MainController(display=display)
        controller.db_connected = True

        controller.handle_ui_action("open_piece_number_dialog")

        self.assertTrue(display.show_piece_number_dialog)
        self.assertEqual(display.piece_number_dialog_input, "3")
        self.assertTrue(display.piece_number_dialog_replace_on_input)

    def test_submit_piece_number_dialog_moves_and_closes_on_success(self):
        display = _DisplayPieceStub()
        controller = MainController(display=display)
        controller.db_connected = True
        controller._load_historic_index = MagicMock(return_value=list(display.historic_images))
        controller._refresh_historic_index_async = MagicMock()
        display.show_piece_number_dialog = True
        display.piece_number_dialog_input = "4"

        controller.handle_ui_action("submit_piece_number_dialog")

        self.assertEqual(display.historic_offset, 0)
        self.assertFalse(display.show_piece_number_dialog)
        self.assertEqual(display.piece_number_dialog_input, "")

    def test_invalid_piece_number_shows_dialog_and_keeps_position(self):
        display = _DisplayPieceStub()
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(return_value=list(display.historic_images))
        original_offset = display.historic_offset

        moved = controller.go_to_historic_piece_number(9, show_missing_dialog=True)

        self.assertFalse(moved)
        self.assertEqual(display.historic_offset, original_offset)
        self.assertTrue(display.show_no_images_dialog)
        self.assertEqual(display.no_images_dialog_message, "Piece 9 not available")


if __name__ == "__main__":
    unittest.main(verbosity=2)

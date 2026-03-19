import unittest
from unittest.mock import MagicMock

from main_controller import MainController


class _DisplaySearchStub:
    def __init__(self):
        self.db = MagicMock()
        self.sftp_client = None
        self.sftp_credentials = None
        self.search_jsn = ""
        self.search_active = True
        self.filtered_suggestions = []
        self.selected_suggestion_idx = -1
        self.available_jsns = ["11861", "1186177", "22700"]
        self.historic_images = []
        self.historic_offset = 0
        self.temp_results = {}
        self._db_result_cache = {}
        self._db_registered_images = set()
        self._historic_index_cache = None
        self._historic_index_mtime = None
        self._historic_index_last_scan = 0.0
        self._historic_jsn_cache = []
        self._image_cache = {}


class TestMainControllerSearchPaste(unittest.TestCase):
    def test_search_paste_replaces_search_box_with_sanitized_digits(self):
        display = _DisplaySearchStub()
        controller = MainController(display=display)

        controller.handle_ui_action("search_paste", text="JSN 11861-77")

        self.assertEqual(display.search_jsn, "1186177")
        self.assertEqual(display.filtered_suggestions, ["1186177"])
        self.assertEqual(display.selected_suggestion_idx, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

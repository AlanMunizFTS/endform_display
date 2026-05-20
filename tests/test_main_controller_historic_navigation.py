import unittest
from unittest.mock import MagicMock

from main_controller import MainController


class TestMainControllerHistoricNavigation(unittest.TestCase):
    def _build_display(self):
        display = MagicMock()
        display.db = MagicMock()
        display.historic_images = []
        display.historic_offset = 0
        display.historic_mode = False
        display.historic_filter_kind = ""
        display.historic_filter_label = ""
        display.historic_filter_jsns = []
        display.historic_filter_total_count = 0
        display.search_active = False
        display.filtered_suggestions = []
        display.selected_suggestion_idx = -1
        display.show_stats_class_modal = False
        display.show_no_images_dialog = False
        display.no_images_dialog_message = ""
        display.temp_results = {}
        display._db_result_cache = {}
        display.set_db_connection = MagicMock()
        return display

    def test_prev_historic_batch_wraps_from_first_to_last(self):
        display = self._build_display()
        display.historic_images = [["batch_1"], ["batch_2"], ["batch_3"]]
        display.historic_offset = 0

        controller = MainController(display=display)

        controller.prev_historic_batch()

        self.assertEqual(display.historic_offset, 2)

    def test_next_historic_batch_wraps_from_last_to_first(self):
        display = self._build_display()
        display.historic_images = [["batch_1"], ["batch_2"], ["batch_3"]]
        display.historic_offset = 2

        controller = MainController(display=display)

        controller.next_historic_batch()

        self.assertEqual(display.historic_offset, 0)

    def test_go_to_historic_jsn_filtered_opens_only_category_subset(self):
        display = self._build_display()
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[
                ["1004_side_cam1.png"],
                ["1003_side_cam1.png"],
                ["1002_side_cam1.png"],
                ["1001_side_cam1.png"],
            ]
        )
        controller._refresh_historic_index_async = MagicMock()

        opened = controller.go_to_historic_jsn_filtered(
            "1002",
            "status",
            "FNOK",
            [{"jsn": "1003"}, {"jsn": "1002"}],
            show_missing_dialog=True,
        )

        self.assertTrue(opened)
        self.assertEqual(display.historic_images, [["1003_side_cam1.png"], ["1002_side_cam1.png"]])
        self.assertEqual(display.historic_offset, 1)
        self.assertEqual(display.historic_filter_kind, "status")
        self.assertEqual(display.historic_filter_label, "FNOK")
        self.assertEqual(display.historic_filter_jsns, ["1003", "1002"])
        self.assertEqual(display.historic_filter_total_count, 2)
        self.assertTrue(display.historic_mode)
        controller._refresh_historic_index_async.assert_called_once_with()

    def test_go_to_historic_jsn_filtered_keeps_total_filter_count_when_some_jsns_are_missing_locally(self):
        display = self._build_display()
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[["1003_side_cam1.png"], ["1002_side_cam1.png"]]
        )
        controller._refresh_historic_index_async = MagicMock()

        opened = controller.go_to_historic_jsn_filtered(
            "1002",
            "status",
            "FNOK",
            [{"jsn": "1004"}, {"jsn": "1003"}, {"jsn": "1002"}],
            show_missing_dialog=True,
        )

        self.assertTrue(opened)
        self.assertEqual(display.historic_images, [["1003_side_cam1.png"], ["1002_side_cam1.png"]])
        self.assertEqual(display.historic_filter_jsns, ["1004", "1003", "1002"])
        self.assertEqual(display.historic_filter_total_count, 3)

    def test_enter_historic_mode_preserves_active_filter(self):
        display = self._build_display()
        display.historic_mode = True
        display.historic_images = [["1003_side_cam1.png"], ["1002_side_cam1.png"]]
        display.historic_offset = 1
        display.historic_filter_kind = "status"
        display.historic_filter_label = "FNOK"
        display.historic_filter_jsns = ["1003", "1002"]
        display.historic_filter_total_count = 2
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[
                ["1004_side_cam1.png"],
                ["1003_side_cam1.png"],
                ["1002_side_cam1.png"],
                ["1001_side_cam1.png"],
            ]
        )
        controller._refresh_historic_index_async = MagicMock()

        controller.enter_historic_mode()

        self.assertEqual(display.historic_images, [["1003_side_cam1.png"], ["1002_side_cam1.png"]])
        self.assertEqual(display.historic_offset, 1)
        self.assertEqual(display.historic_filter_label, "FNOK")

    def test_global_historic_entry_clears_active_filter(self):
        display = self._build_display()
        display.historic_filter_kind = "status"
        display.historic_filter_label = "FNOK"
        display.historic_filter_jsns = ["1003", "1002"]
        display.historic_filter_total_count = 2
        controller = MainController(display=display)
        controller.db_connected = True
        controller.enter_historic_mode = MagicMock()

        controller.handle_ui_action("enter_historic_mode")

        self.assertEqual(display.historic_filter_kind, "")
        self.assertEqual(display.historic_filter_label, "")
        self.assertEqual(display.historic_filter_jsns, [])
        self.assertEqual(display.historic_filter_total_count, 0)
        controller.enter_historic_mode.assert_called_once_with()

    def test_global_jsn_search_clears_active_filter(self):
        display = self._build_display()
        display.historic_filter_kind = "status"
        display.historic_filter_label = "FNOK"
        display.historic_filter_jsns = ["1003", "1002"]
        display.historic_filter_total_count = 2
        controller = MainController(display=display)
        controller._load_historic_index = MagicMock(
            return_value=[["1004_side_cam1.png"], ["1002_side_cam1.png"]]
        )
        controller._refresh_historic_index_async = MagicMock()

        opened = controller.go_to_historic_jsn("1004", show_missing_dialog=True)

        self.assertTrue(opened)
        self.assertEqual(display.historic_images, [["1004_side_cam1.png"], ["1002_side_cam1.png"]])
        self.assertEqual(display.historic_offset, 0)
        self.assertEqual(display.historic_filter_kind, "")
        self.assertEqual(display.historic_filter_label, "")
        self.assertEqual(display.historic_filter_jsns, [])
        self.assertEqual(display.historic_filter_total_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

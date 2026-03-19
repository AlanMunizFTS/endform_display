import unittest
from unittest.mock import MagicMock

from main_controller import MainController


class TestMainControllerHistoricNavigation(unittest.TestCase):
    def _build_display(self):
        display = MagicMock()
        display.db = MagicMock()
        display.historic_images = []
        display.historic_offset = 0
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

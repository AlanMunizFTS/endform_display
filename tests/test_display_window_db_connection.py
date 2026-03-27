import unittest
from unittest.mock import MagicMock, patch

from display_window import DisplayWindow


class TestDisplayWindowDbConnection(unittest.TestCase):
    @patch("display_window.get_db_connection")
    def test_constructor_opens_db_connection_once(self, mock_get_db_connection):
        fake_db = MagicMock()
        mock_get_db_connection.return_value = fake_db

        display = DisplayWindow(file_manager=MagicMock())

        self.assertIs(display.db, fake_db)
        mock_get_db_connection.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)

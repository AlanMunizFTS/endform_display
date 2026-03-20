import unittest
from queue import Queue
from unittest.mock import MagicMock

from main_controller import MainController


class _DisplayRemoteInputStub:
    def __init__(self):
        self.db = MagicMock()
        self.sftp_client = None
        self.sftp_credentials = None
        self.capture_modal_visible = False
        self.capture_modal_text = ""
        self.capture_modal_color = None
        self.capture_modal_expires_at = 0.0


class TestMainControllerRemoteInput(unittest.TestCase):
    def test_send_remote_input_enqueues_trigger_and_shows_capturing_modal(self):
        display = _DisplayRemoteInputStub()
        controller = MainController(display=display)
        controller.stdin_queue = Queue()

        controller.handle_ui_action("send_remote_input")

        self.assertEqual(controller.stdin_queue.get_nowait(), "t\n")
        self.assertTrue(display.capture_modal_visible)
        self.assertEqual(display.capture_modal_text, "capturing")
        self.assertEqual(display.capture_modal_color, (0, 0, 200))
        self.assertEqual(display.capture_modal_expires_at, 0.0)

    def test_send_remote_input_without_stdin_queue_does_not_show_modal(self):
        display = _DisplayRemoteInputStub()
        controller = MainController(display=display)

        controller.handle_ui_action("send_remote_input")

        self.assertFalse(display.capture_modal_visible)
        self.assertEqual(display.capture_modal_text, "")
        self.assertEqual(display.capture_modal_expires_at, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
from unittest.mock import MagicMock

from main_controller import process_remote_event


class _DisplayStub:
    def __init__(self):
        self.trigger_active = False


class TestMainControllerRemoteEvents(unittest.TestCase):
    def test_stdout_trigger_detection_remains_active_without_logging_raw_output(self):
        display = _DisplayStub()
        logger = MagicMock()

        process_remote_event(
            {"type": "stdout", "line": "Waiting for Trigger"},
            display,
            logger,
        )

        self.assertTrue(display.trigger_active)
        logger.info.assert_called_once_with(
            "[REMOTE] Trigger status: ACTIVATED (found 'Waiting for Trigger')",
            allow_repeat=True,
        )

    def test_stderr_output_is_ignored(self):
        display = _DisplayStub()
        logger = MagicMock()

        process_remote_event(
            {"type": "stderr", "line": "sample error"},
            display,
            logger,
        )

        logger.info.assert_not_called()
        logger.warn.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

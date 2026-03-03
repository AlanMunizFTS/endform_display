import unittest
from unittest.mock import MagicMock, patch

import main


class TestMainSFTPFallback(unittest.TestCase):
    def _build_display_mock(self):
        display = MagicMock()
        display.exit_requested = True
        display.historic_mode = False
        display.remote_action_request = None
        display.remote_requested = False
        display.trigger_active = False
        display.connected_cameras = set()
        display.download_historic_batch.return_value = []
        display.file_manager = MagicMock()
        return display

    @patch("main._download_live_images_local", return_value=[])
    @patch("main.get_optional_sftp_settings", return_value=None)
    @patch("main.install_print_logger")
    @patch("main.get_logger")
    @patch("display_window.DisplayWindow")
    @patch("sftp_app.SFTPApp")
    def test_main_local_only_when_optional_settings_missing(
        self,
        sftp_cls,
        display_cls,
        logger_factory,
        _install_logger,
        _optional_settings,
        _download_local,
    ):
        display = self._build_display_mock()
        display_cls.return_value = display
        logger_factory.return_value = MagicMock()

        main.main()

        sftp_cls.assert_not_called()
        kwargs = display_cls.call_args.kwargs
        self.assertIsNone(kwargs["sftp_client"])
        self.assertIsNone(kwargs["sftp_credentials"])
        display.start_historic_download_on_startup.assert_called_once()

    @patch("main._download_live_images_local", return_value=[])
    @patch("main.get_optional_sftp_settings")
    @patch("main.install_print_logger")
    @patch("main.get_logger")
    @patch("display_window.DisplayWindow")
    @patch("sftp_app.SFTPApp")
    def test_main_local_fallback_when_initial_connect_fails(
        self,
        sftp_cls,
        display_cls,
        logger_factory,
        _install_logger,
        optional_settings,
        _download_local,
    ):
        creds = {
            "hostname": "host",
            "port": 22,
            "username": "user",
            "password": "pwd",
        }
        optional_settings.return_value = creds
        fake_sftp = MagicMock()
        fake_sftp.connect_sftp.return_value = False
        fake_sftp.sftp_client = None
        sftp_cls.return_value = fake_sftp
        display = self._build_display_mock()
        display_cls.return_value = display
        logger_factory.return_value = MagicMock()

        main.main()

        sftp_cls.assert_called_once_with("host", 22, "user", "pwd")
        kwargs = display_cls.call_args.kwargs
        self.assertIsNone(kwargs["sftp_client"])
        self.assertEqual(kwargs["sftp_credentials"], creds)
        fake_sftp.disconnect_sftp.assert_called_once()

    @patch("main._download_live_images_remote", return_value=[])
    @patch("main._download_live_images_local", return_value=[])
    @patch("main.get_optional_sftp_settings")
    @patch("main.install_print_logger")
    @patch("main.get_logger")
    @patch("display_window.DisplayWindow")
    @patch("sftp_app.SFTPApp")
    def test_main_uses_sftp_client_when_initial_connect_succeeds(
        self,
        sftp_cls,
        display_cls,
        logger_factory,
        _install_logger,
        optional_settings,
        _download_local,
        _download_remote,
    ):
        creds = {
            "hostname": "host",
            "port": 22,
            "username": "user",
            "password": "pwd",
        }
        optional_settings.return_value = creds
        fake_sftp = MagicMock()
        fake_sftp.connect_sftp.return_value = True
        fake_sftp.sftp_client = object()
        sftp_cls.return_value = fake_sftp
        display = self._build_display_mock()
        display_cls.return_value = display
        logger_factory.return_value = MagicMock()

        main.main()

        kwargs = display_cls.call_args.kwargs
        self.assertIs(kwargs["sftp_client"], fake_sftp.sftp_client)
        self.assertEqual(kwargs["sftp_credentials"], creds)
        fake_sftp.disconnect_sftp.assert_called_once()

    @patch("main.time.monotonic")
    @patch("main._download_live_images_local", return_value=[])
    @patch("main.get_optional_sftp_settings")
    @patch("main.install_print_logger")
    @patch("main.get_logger")
    @patch("display_window.DisplayWindow")
    @patch("sftp_app.SFTPApp")
    def test_main_periodic_retry_attempts_reconnect(
        self,
        sftp_cls,
        display_cls,
        logger_factory,
        _install_logger,
        optional_settings,
        _download_local,
        monotonic_mock,
    ):
        creds = {
            "hostname": "host",
            "port": 22,
            "username": "user",
            "password": "pwd",
        }
        optional_settings.return_value = creds
        fake_sftp = MagicMock()
        fake_sftp.connect_sftp.side_effect = [False, True]
        fake_sftp.sftp_client = object()
        sftp_cls.return_value = fake_sftp
        display = self._build_display_mock()
        display.exit_requested = False
        display.show_image_grid.side_effect = lambda *args, **kwargs: setattr(display, "exit_requested", True)
        display_cls.return_value = display
        logger_factory.return_value = MagicMock()
        monotonic_mock.side_effect = [0.0, 11.0, 11.0]

        main.main()

        self.assertEqual(fake_sftp.connect_sftp.call_count, 2)
        display.set_sftp_client.assert_any_call(fake_sftp.sftp_client)

    @patch("main.time.monotonic")
    @patch("main._download_live_images_local", return_value=[])
    @patch("main.get_optional_sftp_settings")
    @patch("main.install_print_logger")
    @patch("main.get_logger")
    @patch("display_window.DisplayWindow")
    @patch("sftp_app.SFTPApp")
    def test_main_start_action_while_disconnected_attempts_reconnect(
        self,
        sftp_cls,
        display_cls,
        logger_factory,
        _install_logger,
        optional_settings,
        _download_local,
        monotonic_mock,
    ):
        creds = {
            "hostname": "host",
            "port": 22,
            "username": "user",
            "password": "pwd",
        }
        optional_settings.return_value = creds
        fake_sftp = MagicMock()
        fake_sftp.connect_sftp.side_effect = [False, False]
        fake_sftp.sftp_client = None
        sftp_cls.return_value = fake_sftp
        display = self._build_display_mock()
        display.exit_requested = False
        display.remote_action_request = "start"
        display.show_image_grid.side_effect = lambda *args, **kwargs: setattr(display, "exit_requested", True)
        display_cls.return_value = display
        logger_factory.return_value = MagicMock()
        monotonic_mock.side_effect = [0.0, 0.0, 0.0, 0.0, 0.0]

        main.main()

        self.assertEqual(fake_sftp.connect_sftp.call_count, 2)
        fake_sftp.start_remote_process_multiprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

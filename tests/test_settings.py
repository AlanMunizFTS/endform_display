import os
import unittest
from unittest.mock import patch

import settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        self._env_loaded_original = settings._ENV_LOADED
        settings._ENV_LOADED = True

    def tearDown(self):
        settings._ENV_LOADED = self._env_loaded_original

    def test_is_sftp_enabled_truthy_values(self):
        for value in ("1", "true", "True", "yes", "on"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"APP_SFTP_ENABLED": value}, clear=True):
                    self.assertTrue(settings.is_sftp_enabled())

    def test_is_sftp_enabled_false_when_missing_or_falsey(self):
        falsey_values = [None, "", "0", "false", "off", "no"]
        for value in falsey_values:
            with self.subTest(value=value):
                env = {}
                if value is not None:
                    env["APP_SFTP_ENABLED"] = value
                with patch.dict(os.environ, env, clear=True):
                    self.assertFalse(settings.is_sftp_enabled())

    def test_get_optional_sftp_settings_returns_none_when_disabled(self):
        with patch.dict(
            os.environ,
            {
                "APP_SFTP_ENABLED": "0",
                "SFTP_HOST": "host",
                "SFTP_PORT": "22",
                "SFTP_USERNAME": "user",
                "SFTP_PASSWORD": "pwd",
            },
            clear=True,
        ):
            self.assertIsNone(settings.get_optional_sftp_settings())

    def test_get_optional_sftp_settings_returns_none_when_missing_creds(self):
        with patch.dict(
            os.environ,
            {"APP_SFTP_ENABLED": "1", "SFTP_HOST": "host", "SFTP_PORT": "22"},
            clear=True,
        ):
            self.assertIsNone(settings.get_optional_sftp_settings())

    def test_get_optional_sftp_settings_raises_for_invalid_port(self):
        with patch.dict(
            os.environ,
            {
                "APP_SFTP_ENABLED": "1",
                "SFTP_HOST": "host",
                "SFTP_PORT": "not-an-int",
                "SFTP_USERNAME": "user",
                "SFTP_PASSWORD": "pwd",
            },
            clear=True,
        ):
            with self.assertRaises(RuntimeError):
                settings.get_optional_sftp_settings()

    def test_get_optional_sftp_settings_returns_valid_dict(self):
        with patch.dict(
            os.environ,
            {
                "APP_SFTP_ENABLED": "1",
                "SFTP_HOST": "host",
                "SFTP_PORT": "22",
                "SFTP_USERNAME": "user",
                "SFTP_PASSWORD": "pwd",
            },
            clear=True,
        ):
            self.assertEqual(
                settings.get_optional_sftp_settings(),
                {
                    "hostname": "host",
                    "port": 22,
                    "username": "user",
                    "password": "pwd",
                },
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

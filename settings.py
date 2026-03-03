import os
from pathlib import Path


_ENV_LOADED = False


def load_env_file(env_path=".env"):
    """Load key=value pairs from a local .env file into os.environ once."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    path = Path(env_path)
    if not path.exists():
        _ENV_LOADED = True
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def _get_required_env(name):
    load_env_file()
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_optional_env(name):
    load_env_file()
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value


def _get_required_int_env(name):
    raw_value = _get_required_env(name)
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name} must be an integer. Got: {raw_value!r}"
        ) from exc


def get_sftp_settings():
    """Return validated SFTP settings from environment variables."""
    return {
        "hostname": _get_required_env("SFTP_HOST"),
        "port": _get_required_int_env("SFTP_PORT"),
        "username": _get_required_env("SFTP_USERNAME"),
        "password": _get_required_env("SFTP_PASSWORD"),
    }


def is_sftp_enabled():
    """Return whether SFTP mode is explicitly enabled by environment flag."""
    load_env_file()
    raw_value = os.getenv("APP_SFTP_ENABLED")
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_optional_sftp_settings():
    """Return validated SFTP settings when enabled and complete, else None."""
    if not is_sftp_enabled():
        return None

    hostname = _get_optional_env("SFTP_HOST")
    port_raw = _get_optional_env("SFTP_PORT")
    username = _get_optional_env("SFTP_USERNAME")
    password = _get_optional_env("SFTP_PASSWORD")

    if not all([hostname, port_raw, username, password]):
        return None

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable SFTP_PORT must be an integer. Got: {port_raw!r}"
        ) from exc

    return {
        "hostname": hostname,
        "port": port,
        "username": username,
        "password": password,
    }


def get_db_settings():
    """Return validated database settings from environment variables."""
    return {
        "host": _get_required_env("DB_HOST"),
        "port": _get_required_int_env("DB_PORT"),
        "database": _get_required_env("DB_NAME"),
        "user": _get_required_env("DB_USER"),
        "password": _get_required_env("DB_PASSWORD"),
    }

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

import paramiko

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings import get_sftp_settings, load_env_file


DEFAULT_CONTAINER_NAME = "postgres-vision"
DEFAULT_DB_NAME = "postgres"
DEFAULT_DB_USER = "postgres"
STREAM_CHUNK_SIZE = 64 * 1024
DEFAULT_TIMEOUT = 30
POLL_INTERVAL_SECONDS = 0.05


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Conecta por SSH al servidor remoto y transmite un pg_dump de Postgres "
            "ejecutado dentro de un contenedor Docker."
        )
    )
    parser.add_argument("--host", help="Host SSH remoto. Default: SFTP_HOST del .env")
    parser.add_argument(
        "--port",
        type=int,
        help="Puerto SSH remoto. Default: SFTP_PORT del .env",
    )
    parser.add_argument(
        "--ssh-user",
        help="Usuario SSH remoto. Default: SFTP_USERNAME del .env",
    )
    parser.add_argument(
        "--ssh-password",
        help="Password SSH remoto. Default: SFTP_PASSWORD del .env",
    )
    parser.add_argument(
        "--container",
        default=DEFAULT_CONTAINER_NAME,
        help="Nombre del contenedor Docker remoto. Default: %(default)s",
    )
    parser.add_argument(
        "--database",
        help="Base de datos a exportar. Default: DB_NAME del .env o postgres",
    )
    parser.add_argument(
        "--db-user",
        help="Usuario de Postgres para pg_dump. Default: DB_USER del .env o postgres",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ruta local opcional donde guardar el dump en lugar de stdout",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout SSH en segundos para conectar y lanzar el comando. Default: %(default)s",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Ejecuta docker sin sudo en el servidor remoto",
    )
    return parser.parse_args(argv)


def _get_env_default(name, fallback):
    load_env_file()
    value = os.getenv(name)
    if value is None:
        return fallback
    value = value.strip()
    return value or fallback


def resolve_config(args):
    ssh_settings = get_sftp_settings()
    return {
        "host": args.host or ssh_settings["hostname"],
        "port": args.port or ssh_settings["port"],
        "ssh_user": args.ssh_user or ssh_settings["username"],
        "ssh_password": args.ssh_password or ssh_settings["password"],
        "container": args.container or DEFAULT_CONTAINER_NAME,
        "database": args.database or _get_env_default("DB_NAME", DEFAULT_DB_NAME),
        "db_user": args.db_user or _get_env_default("DB_USER", DEFAULT_DB_USER),
        "output": args.output,
        "timeout": args.timeout,
        "use_sudo": not args.no_sudo,
    }


def build_remote_dump_command(container, database, db_user, use_sudo=True):
    command_parts = [
        "docker",
        "exec",
        "-i",
        container,
        "pg_dump",
        "-U",
        db_user,
        "-d",
        database,
        "--format=plain",
        "--no-owner",
        "--no-privileges",
    ]
    quoted = " ".join(shlex.quote(part) for part in command_parts)
    if use_sudo:
        return f"sudo -S -p '' {quoted}"
    return quoted


def _write_status(message):
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def stream_command_output(channel, output_stream, error_stream):
    while True:
        made_progress = False

        if channel.recv_ready():
            chunk = channel.recv(STREAM_CHUNK_SIZE)
            if chunk:
                output_stream.write(chunk)
                output_stream.flush()
                made_progress = True

        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(STREAM_CHUNK_SIZE)
            if chunk:
                error_stream.write(chunk)
                error_stream.flush()
                made_progress = True

        if channel.exit_status_ready():
            if not channel.recv_ready() and not channel.recv_stderr_ready():
                break

        if not made_progress:
            time.sleep(POLL_INTERVAL_SECONDS)

    return channel.recv_exit_status()


def export_remote_db(config, ssh_client_factory=None):
    ssh_client_factory = ssh_client_factory or paramiko.SSHClient
    client = ssh_client_factory()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    output_path = Path(config["output"]).resolve() if config["output"] else None
    output_stream = None
    output_target = None

    try:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_stream = output_path.open("wb")
            output_target = output_stream
            _write_status(f"Writing remote dump to {output_path}")
        else:
            output_target = sys.stdout.buffer

        _write_status(
            f"Connecting to {config['host']}:{config['port']} as {config['ssh_user']}"
        )
        client.connect(
            hostname=config["host"],
            port=config["port"],
            username=config["ssh_user"],
            password=config["ssh_password"],
            timeout=config["timeout"],
        )

        command = build_remote_dump_command(
            container=config["container"],
            database=config["database"],
            db_user=config["db_user"],
            use_sudo=config["use_sudo"],
        )
        _write_status(
            f"Starting pg_dump for database {config['database']} from container {config['container']}"
        )
        stdin, stdout, _stderr = client.exec_command(
            command,
            timeout=config["timeout"],
            get_pty=False,
        )
        if config["use_sudo"]:
            stdin.write(f"{config['ssh_password']}\n")
            stdin.flush()
        stdin.close()

        exit_code = stream_command_output(
            stdout.channel,
            output_target,
            sys.stderr.buffer,
        )
        if exit_code != 0:
            raise RuntimeError(f"Remote pg_dump failed with exit code {exit_code}")

        _write_status("Remote pg_dump completed successfully")
        return 0
    finally:
        if output_stream is not None:
            output_stream.close()
        try:
            client.close()
        except Exception:
            pass


def main(argv=None):
    try:
        args = parse_args(argv)
        config = resolve_config(args)
        return export_remote_db(config)
    except KeyboardInterrupt:
        _write_status("Export cancelled by user")
        return 130
    except Exception as exc:
        _write_status(f"Export failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utilities.export_remote_db as export_remote_db


class _FakeChannel:
    def __init__(self, stdout_chunks=None, stderr_chunks=None, exit_code=0):
        self.stdout_chunks = list(stdout_chunks or [])
        self.stderr_chunks = list(stderr_chunks or [])
        self.exit_code = exit_code

    def recv_ready(self):
        return bool(self.stdout_chunks)

    def recv(self, _size):
        return self.stdout_chunks.pop(0) if self.stdout_chunks else b""

    def recv_stderr_ready(self):
        return bool(self.stderr_chunks)

    def recv_stderr(self, _size):
        return self.stderr_chunks.pop(0) if self.stderr_chunks else b""

    def exit_status_ready(self):
        return not self.stdout_chunks and not self.stderr_chunks

    def recv_exit_status(self):
        return self.exit_code


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.flushed = False
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class _FakeSSHClient:
    def __init__(self, channel=None):
        self.channel = channel or _FakeChannel()
        self.policy = None
        self.connect_kwargs = None
        self.exec_calls = []
        self.stdin = _FakeStdin()
        self.closed = False

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kwargs):
        self.connect_kwargs = kwargs

    def exec_command(self, command, timeout=None, get_pty=False):
        self.exec_calls.append(
            {
                "command": command,
                "timeout": timeout,
                "get_pty": get_pty,
            }
        )
        stdout = SimpleNamespace(channel=self.channel)
        stderr = SimpleNamespace(channel=self.channel)
        return self.stdin, stdout, stderr

    def close(self):
        self.closed = True


class _BinaryRecorder:
    def __init__(self):
        self.chunks = []

    def write(self, data):
        self.chunks.append(data)

    def flush(self):
        pass

    def getvalue(self):
        return b"".join(self.chunks)


class TestExportRemoteDb(unittest.TestCase):
    def test_build_remote_dump_command_defaults_to_sudo(self):
        command = export_remote_db.build_remote_dump_command(
            container="postgres-vision",
            database="postgres",
            db_user="postgres",
            use_sudo=True,
        )

        self.assertEqual(
            command,
            "sudo -S -p '' docker exec -i postgres-vision pg_dump -U postgres -d postgres --format=plain --no-owner --no-privileges",
        )

    def test_resolve_config_uses_env_defaults_and_sftp_settings(self):
        args = export_remote_db.parse_args([])
        with patch.object(
            export_remote_db,
            "get_sftp_settings",
            return_value={
                "hostname": "10.0.0.5",
                "port": 22,
                "username": "vision",
                "password": "secret",
            },
        ), patch.object(export_remote_db, "load_env_file"), patch.dict(
            os.environ,
            {"DB_NAME": "postgres", "DB_USER": "postgres"},
            clear=True,
        ):
            config = export_remote_db.resolve_config(args)

        self.assertEqual(config["host"], "10.0.0.5")
        self.assertEqual(config["port"], 22)
        self.assertEqual(config["ssh_user"], "vision")
        self.assertEqual(config["ssh_password"], "secret")
        self.assertEqual(config["database"], "postgres")
        self.assertEqual(config["db_user"], "postgres")
        self.assertTrue(config["use_sudo"])

    def test_export_remote_db_streams_chunks_to_stdout_and_stderr(self):
        fake_channel = _FakeChannel(
            stdout_chunks=[b"line 1\n", b"line 2\n"],
            stderr_chunks=[b"notice\n"],
            exit_code=0,
        )
        fake_client = _FakeSSHClient(channel=fake_channel)
        fake_stdout = _BinaryRecorder()
        fake_stderr = _BinaryRecorder()
        text_stderr = io.StringIO()
        config = {
            "host": "host",
            "port": 22,
            "ssh_user": "vision",
            "ssh_password": "secret",
            "container": "postgres-vision",
            "database": "postgres",
            "db_user": "postgres",
            "output": None,
            "timeout": 30,
            "use_sudo": True,
        }

        with patch.object(sys, "stdout", SimpleNamespace(buffer=fake_stdout)), patch.object(
            sys,
            "stderr",
            SimpleNamespace(buffer=fake_stderr, write=text_stderr.write, flush=text_stderr.flush),
        ), patch.object(export_remote_db.paramiko, "AutoAddPolicy", return_value=object()):
            result = export_remote_db.export_remote_db(
                config,
                ssh_client_factory=lambda: fake_client,
            )

        self.assertEqual(result, 0)
        self.assertEqual(fake_stdout.getvalue(), b"line 1\nline 2\n")
        self.assertIn(b"notice\n", fake_stderr.getvalue())
        self.assertEqual(fake_client.stdin.writes, ["secret\n"])
        self.assertTrue(fake_client.stdin.closed)
        self.assertTrue(fake_client.closed)
        self.assertEqual(
            fake_client.exec_calls[0]["command"],
            "sudo -S -p '' docker exec -i postgres-vision pg_dump -U postgres -d postgres --format=plain --no-owner --no-privileges",
        )

    def test_export_remote_db_writes_to_output_file(self):
        fake_channel = _FakeChannel(stdout_chunks=[b"create table test;\n"], exit_code=0)
        fake_client = _FakeSSHClient(channel=fake_channel)
        fake_stderr = _BinaryRecorder()
        text_stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "exports", "remote.sql")
            config = {
                "host": "host",
                "port": 22,
                "ssh_user": "vision",
                "ssh_password": "secret",
                "container": "postgres-vision",
                "database": "postgres",
                "db_user": "postgres",
                "output": output_path,
                "timeout": 30,
                "use_sudo": False,
            }

            with patch.object(
                sys,
                "stderr",
                SimpleNamespace(
                    buffer=fake_stderr,
                    write=text_stderr.write,
                    flush=text_stderr.flush,
                ),
            ), patch.object(export_remote_db.paramiko, "AutoAddPolicy", return_value=object()):
                result = export_remote_db.export_remote_db(
                    config,
                    ssh_client_factory=lambda: fake_client,
                )

            self.assertEqual(result, 0)
            with open(output_path, "rb") as handle:
                self.assertEqual(handle.read(), b"create table test;\n")
            self.assertEqual(fake_client.stdin.writes, [])
            self.assertIn("Writing remote dump to", text_stderr.getvalue())

    def test_export_remote_db_raises_on_non_zero_exit(self):
        fake_channel = _FakeChannel(stderr_chunks=[b"permission denied\n"], exit_code=1)
        fake_client = _FakeSSHClient(channel=fake_channel)
        fake_stderr = _BinaryRecorder()
        text_stderr = io.StringIO()
        config = {
            "host": "host",
            "port": 22,
            "ssh_user": "vision",
            "ssh_password": "secret",
            "container": "postgres-vision",
            "database": "postgres",
            "db_user": "postgres",
            "output": None,
            "timeout": 30,
            "use_sudo": True,
        }

        with patch.object(sys, "stdout", SimpleNamespace(buffer=_BinaryRecorder())), patch.object(
            sys,
            "stderr",
            SimpleNamespace(buffer=fake_stderr, write=text_stderr.write, flush=text_stderr.flush),
        ), patch.object(export_remote_db.paramiko, "AutoAddPolicy", return_value=object()):
            with self.assertRaises(RuntimeError):
                export_remote_db.export_remote_db(
                    config,
                    ssh_client_factory=lambda: fake_client,
                )

        self.assertIn(b"permission denied\n", fake_stderr.getvalue())

    def test_main_returns_non_zero_on_failure(self):
        with patch.object(
            export_remote_db,
            "resolve_config",
            return_value={},
        ), patch.object(
            export_remote_db,
            "export_remote_db",
            side_effect=RuntimeError("boom"),
        ), patch.object(
            export_remote_db,
            "parse_args",
            return_value=SimpleNamespace(),
        ), patch.object(
            sys,
            "stderr",
            SimpleNamespace(
                buffer=_BinaryRecorder(),
                write=lambda text: None,
                flush=lambda: None,
            ),
        ):
            self.assertEqual(export_remote_db.main([]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

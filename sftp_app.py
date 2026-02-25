import paramiko

import threading
import time
import posixpath
from collections import deque
from file_manager import FileManager
from utilities.log import get_logger
from settings import get_sftp_settings

logger = get_logger()


def _safe_put_nowait(queue_obj, payload):
    if queue_obj is None:
        return
    try:
        queue_obj.put_nowait(payload)
    except Exception:
        pass


def _remote_process_worker(hostname, port, username, password, command, pid_queue=None, stop_event=None, status_queue=None):
    app = SFTPApp(hostname, port, username, password)
    if not app.connect_sftp():
        return
    try:
        print("[REMOTE] Start execution")
        app.start_remote_process(command)
        pid_sent = False
        last_out_idx = 0
        last_err_idx = 0
        while True:
            try:
                if not pid_sent and app.remote_process and app.remote_process.pid:
                    if pid_queue is not None:
                        try:
                            pid_queue.put(app.remote_process.pid)
                        except Exception:
                            pass
                    _safe_put_nowait(
                        status_queue,
                        {"type": "pid", "pid": str(app.remote_process.pid)},
                    )
                    pid_sent = True
                if stop_event is not None and stop_event.is_set():
                    break
                if app.remote_process:
                    output_lines = app.remote_process.get_output()
                    if last_out_idx < len(output_lines):
                        for line in output_lines[last_out_idx:]:
                            _safe_put_nowait(status_queue, {"type": "stdout", "line": line})
                        last_out_idx = len(output_lines)

                    error_lines = app.remote_process.get_errors()
                    if last_err_idx < len(error_lines):
                        for line in error_lines[last_err_idx:]:
                            _safe_put_nowait(status_queue, {"type": "stderr", "line": line})
                        last_err_idx = len(error_lines)
                time.sleep(0.2)
            except KeyboardInterrupt:
                break
    finally:
        logger.info("[REMOTE] End execution")
        _safe_put_nowait(status_queue, {"type": "worker_stopped"})
        app.disconnect_sftp()


class RemoteProcess:
    def __init__(self, ssh_client, max_lines=None):
        self.ssh_client = ssh_client
        self.channel = None
        self.stdout = None
        self.stderr = None
        self.pid = None
        self._max_lines = max_lines if isinstance(max_lines, int) and max_lines > 0 else None
        if self._max_lines:
            self.lines = deque(maxlen=self._max_lines)
            self.error_lines = deque(maxlen=self._max_lines)
        else:
            # Store all lines when max_lines is not set
            self.lines = []
            self.error_lines = []
        self._reader_thread = None
        self._running = False
        self._stdout_buffer = ""
        self._stderr_buffer = ""

    def start(self, command):
        """Start a remote process and stream stdout/stderr in real time."""
        if not self.ssh_client:
            raise RuntimeError("SSH client is not connected")
        if self._running:
            return

        self._running = True
        self.pid = None
        self.lines.clear()
        self.error_lines.clear()

        self.lines.append(f"[START] {command}")
        self.stdin, self.stdout, self.stderr = self.ssh_client.exec_command(
            command, get_pty=True
        )
        self.channel = self.stdout.channel if self.stdout else None
        self._stdout_buffer = ""
        self._stderr_buffer = ""

        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        """Read stdout/stderr continuously until the process stops."""
        try:
            while self._running and self.channel and not self.channel.exit_status_ready():
                self._drain_channels()
                time.sleep(0.05)

            # Drain any remaining output after exit
            self._drain_channels(final=True)

            if self.channel:
                exit_code = self.channel.recv_exit_status()
                self.lines.append(f"[EXIT] code={exit_code}")
        finally:
            self._running = False

    def _drain_channels(self, final=False):
        if not self.channel:
            return

        if self.channel.recv_ready():
            data = self.channel.recv(4096).decode(errors="replace")
            self._stdout_buffer += data
            self._flush_stdout_lines()
            self._flush_stdout_partial()

        if self.channel.recv_stderr_ready():
            data = self.channel.recv_stderr(4096).decode(errors="replace")
            self._stderr_buffer += data
            self._flush_stderr_lines()
            self._flush_stderr_partial()

        if final:
            self._flush_stdout_lines(final=True)
            self._flush_stderr_lines(final=True)
            self._flush_stdout_partial(final=True)
            self._flush_stderr_partial(final=True)

    def _flush_stdout_lines(self, final=False):
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                if self.pid is None and line.isdigit():
                    self.pid = line
                    self.lines.append(f"PID: {self.pid}")
                else:
                    self.lines.append(line)
        if final and self._stdout_buffer:
            line = self._stdout_buffer.rstrip("\r")
            if line:
                if self.pid is None and line.isdigit():
                    self.pid = line
                    self.lines.append(f"PID: {self.pid}")
                else:
                    self.lines.append(line)
            self._stdout_buffer = ""

    def _flush_stderr_lines(self, final=False):
        while "\n" in self._stderr_buffer:
            line, self._stderr_buffer = self._stderr_buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.error_lines.append(line)
        if final and self._stderr_buffer:
            line = self._stderr_buffer.rstrip("\r")
            if line:
                self.error_lines.append(line)
            self._stderr_buffer = ""

    def _flush_stdout_partial(self, final=False):
        if final:
            return
        if self._stdout_buffer and "\n" not in self._stdout_buffer:
            self.lines.append(self._stdout_buffer.rstrip("\r"))
            self._stdout_buffer = ""

    def _flush_stderr_partial(self, final=False):
        if final:
            return
        if self._stderr_buffer and "\n" not in self._stderr_buffer:
            self.error_lines.append(self._stderr_buffer.rstrip("\r"))
            self._stderr_buffer = ""

    def stop(self):
        """Stop the remote process."""
        if self.pid and self.ssh_client:
            try:
                self.ssh_client.exec_command(f"kill {self.pid}")
            except Exception:
                pass

        try:
            if self.channel:
                self.channel.close()
        except Exception:
            pass

        self._running = False

    def get_output(self):
        """Return recent stdout lines."""
        return list(self.lines)

    def get_errors(self):
        """Return recent stderr lines."""
        return list(self.error_lines)

class SFTPApp:
    def __init__(self, hostname, port, username, password, file_manager=None):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.file_manager = file_manager or FileManager()
        self.sftp_client = None
        self.ssh_client = None
        self.remote_process = None
        
    def log_message(self, message, level="INFO"):
        """Write message to logger with optional level."""
        if level == "ERROR":
            logger.error(message, allow_repeat=True)
        elif level == "WARN":
            logger.warn(message, allow_repeat=True)
        elif level == "DEBUG":
            logger.debug(message, allow_repeat=True)
        else:
            logger.info(message, allow_repeat=True)
        
    def connect_sftp(self):
        """Connect to SFTP server"""
        try:
            self.log_message(
                f"[SSH] Connecting to {self.hostname}:{self.port} as {self.username}"
            )
            # Create SSH client
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect
            self.ssh_client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=10
            )
            
            # Open SFTP session
            self.sftp_client = self.ssh_client.open_sftp()
            self.log_message("[SSH] Connection successful")
            
            return True
            
        except paramiko.AuthenticationException:
            self.log_message("[SSH] Authentication failed", level="ERROR")
            return False
        except paramiko.SSHException as e:
            self.log_message(f"[SSH] SSH error: {str(e)}", level="ERROR")
            return False
        except Exception as e:
            self.log_message(f"[SSH] Connection error: {str(e)}", level="ERROR")
            return False
            
    def start_remote_process(self, command):
        """Start a remote process over the existing SSH connection."""
        if not self.ssh_client:
            self.log_message("[WARN] SSH client not connected")
            return False
        self.remote_process = RemoteProcess(self.ssh_client)
        self.remote_process.start(command)
        return True

    def start_remote_process_multiprocess(self, command, pid_queue=None, stop_event=None, status_queue=None):
        """Start the remote process in a separate local process.

        status_queue receives raw events:
        {"type": "pid"|"stdout"|"stderr"|"worker_stopped", ...}
        """
        from multiprocessing import Process
        proc = Process(
            target=_remote_process_worker,
            args=(self.hostname, self.port, self.username, self.password, command, pid_queue, stop_event, status_queue)
        )
        proc.daemon = True
        proc.start()
        return proc

    def is_pid_running(self, pid):
        """Check if a PID is running on the remote host."""
        if not self.ssh_client or not pid:
            return False
        try:
            _, stdout, _ = self.ssh_client.exec_command(f"ps -p {pid} -o pid=")
            output = stdout.read().decode().strip()
            return output == str(pid)
        except Exception:
            return False

    def list_remote_files(self, remote_dir):
        """List files in a remote directory."""
        if not self.sftp_client:
            raise RuntimeError("SFTP client is not connected")
        return self.file_manager.sftp_listdir(self.sftp_client, remote_dir)

    def download_file(self, remote_path, local_path):
        """Download one file from remote_path to local_path."""
        if not self.sftp_client:
            raise RuntimeError("SFTP client is not connected")
        self.file_manager.sftp_get(self.sftp_client, remote_path, local_path)

    def upload_file(self, local_path, remote_path):
        """Upload one file from local_path to remote_path."""
        if not self.sftp_client:
            raise RuntimeError("SFTP client is not connected")
        self.file_manager.sftp_put(self.sftp_client, local_path, remote_path)

    def ensure_remote_dir(self, remote_dir):
        """Create remote directory if it does not exist."""
        if not self.sftp_client:
            raise RuntimeError("SFTP client is not connected")
        try:
            self.file_manager.sftp_stat(self.sftp_client, remote_dir)
        except FileNotFoundError:
            self.file_manager.sftp_mkdir(self.sftp_client, remote_dir)

    def join_remote_path(self, remote_dir, filename):
        """Join remote dir + filename using POSIX separators."""
        return posixpath.join(remote_dir, filename)

    def disconnect_sftp(self):
        """Disconnect from SFTP server"""
        try:
            if self.remote_process:
                self.remote_process.stop()
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
                
            self.sftp_client = None
            self.ssh_client = None
            self.remote_process = None
            
        except Exception as e:
            self.log_message(f"Error disconnecting: {str(e)}")
            
if __name__ == "__main__":
    # Simple PID inspection test
    sftp_settings = get_sftp_settings()
    hostname = sftp_settings["hostname"]
    port = sftp_settings["port"]
    username = sftp_settings["username"]
    password = sftp_settings["password"]

    app = SFTPApp(hostname, port, username, password)
    if app.connect_sftp():
        try:
            test_pid = "2578997"
            running = app.is_pid_running(test_pid)
            logger.info(f"PID {test_pid} running: {running}")
        finally:
            app.disconnect_sftp()

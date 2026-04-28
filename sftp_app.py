import paramiko

import posixpath
from file_manager import FileManager
from utilities.log import get_logger
from settings import get_sftp_settings

logger = get_logger()


class SFTPApp:
    def __init__(self, hostname, port, username, password, file_manager=None):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.file_manager = file_manager or FileManager()
        self.sftp_client = None
        self.ssh_client = None
        
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
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
                
            self.sftp_client = None
            self.ssh_client = None
            
        except Exception as e:
            self.log_message(f"Error disconnecting: {str(e)}")
            
if __name__ == "__main__":
    # Simple SFTP connectivity check.
    sftp_settings = get_sftp_settings()
    hostname = sftp_settings["hostname"]
    port = sftp_settings["port"]
    username = sftp_settings["username"]
    password = sftp_settings["password"]

    app = SFTPApp(hostname, port, username, password)
    if app.connect_sftp():
        try:
            logger.info("[SSH] SFTP connectivity check succeeded", allow_repeat=True)
        finally:
            app.disconnect_sftp()

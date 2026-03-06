import os
import shutil
import sys
import threading
from typing import Iterable, List, Optional

import cv2


class FileManager:
    """Thin adapter for filesystem and SFTP file operations."""
    _imread_stderr_lock = threading.Lock()

    def join(self, *parts: str) -> str:
        return os.path.join(*parts)

    def basename(self, path: str) -> str:
        return os.path.basename(path)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_file(self, path: str) -> bool:
        return os.path.isfile(path)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def listdir(self, path: str) -> List[str]:
        return os.listdir(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path: str) -> None:
        os.remove(path)

    def rmtree(self, path: str) -> None:
        shutil.rmtree(path)

    def copy2(self, src: str, dst: str) -> None:
        shutil.copy2(src, dst)

    def getmtime(self, path: str) -> float:
        return os.path.getmtime(path)

    def getsize(self, path: str) -> int:
        return os.path.getsize(path)

    def read_image(self, path: str, flags: int = cv2.IMREAD_COLOR):
        # OpenCV/codec backends may print decode errors (e.g., libpng) directly to
        # stderr. Suppress stderr for this call and handle decode failures via `None`.
        stderr_fd = None
        saved_stderr_fd = None
        null_fd = None
        with self._imread_stderr_lock:
            try:
                stderr_fd = sys.stderr.fileno()
            except Exception:
                return cv2.imread(path, flags)

            try:
                saved_stderr_fd = os.dup(stderr_fd)
                null_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(null_fd, stderr_fd)
                return cv2.imread(path, flags)
            finally:
                if saved_stderr_fd is not None:
                    try:
                        os.dup2(saved_stderr_fd, stderr_fd)
                    except Exception:
                        pass
                    try:
                        os.close(saved_stderr_fd)
                    except Exception:
                        pass
                if null_fd is not None:
                    try:
                        os.close(null_fd)
                    except Exception:
                        pass

    def write_image(self, path: str, image) -> bool:
        return bool(cv2.imwrite(path, image))

    def list_files_by_extension(
        self, path: str, extensions: Iterable[str], case_sensitive: bool = False
    ) -> List[str]:
        items = self.listdir(path)
        if case_sensitive:
            valid = tuple(extensions)
            return [name for name in items if name.endswith(valid)]
        valid = tuple(ext.lower() for ext in extensions)
        return [name for name in items if name.lower().endswith(valid)]

    # SFTP wrappers (client is passed by caller; no connection ownership here).
    def sftp_chdir(self, sftp_client, remote_dir: str) -> None:
        sftp_client.chdir(remote_dir)

    def sftp_listdir(self, sftp_client, remote_dir: Optional[str] = None) -> List[str]:
        if remote_dir:
            sftp_client.chdir(remote_dir)
        return sftp_client.listdir()

    def sftp_remove(self, sftp_client, remote_path: str) -> None:
        sftp_client.remove(remote_path)

    def sftp_get(self, sftp_client, remote_path: str, local_path: str) -> None:
        sftp_client.get(remote_path, local_path)

    def sftp_put(self, sftp_client, local_path: str, remote_path: str) -> None:
        sftp_client.put(local_path, remote_path)

    def sftp_stat(self, sftp_client, remote_path: str):
        return sftp_client.stat(remote_path)

    def sftp_mkdir(self, sftp_client, remote_path: str) -> None:
        sftp_client.mkdir(remote_path)

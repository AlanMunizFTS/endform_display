import os
import re
import threading
from datetime import datetime


class Logger:
    def __init__(
        self,
        path="log.txt",
        reset=False,
        dedupe=True,
        normalize_numbers=True,
        min_level="INFO",
    ):
        self.path = path
        self.dedupe = dedupe
        self.normalize_numbers = normalize_numbers
        self._seen = set()
        self._lock = threading.Lock()
        self._levels = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
        self._min_level = self._levels.get(min_level, 20)

        if reset:
            self.reset()

    def _is_excluded(self, message):
        lower_msg = message.lower()
        if "[hist_sync_ssh] downloaded " in lower_msg and "new historic images" in lower_msg:
            return True
        if "background" in lower_msg:
            return True
        if "?" in message:
            return True
        stripped = message.strip()
        if stripped and all(ch == "=" for ch in stripped):
            return True
        return False

    def reset(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._lock:
            with open(self.path, "w", encoding="utf-8"):
                pass
            self._seen.clear()

    def _normalize(self, message):
        msg = message.strip()
        if self.normalize_numbers:
            msg = re.sub(r"\d+", "#", msg)
        return msg

    def _should_log(self, level, message, allow_repeat):
        if self._levels.get(level, 20) < self._min_level:
            return False
        if not self.dedupe or allow_repeat:
            return True
        key = self._normalize(message)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def log(self, message, level="INFO", allow_repeat=False):
        if message is None:
            return
        message = str(message)
        if self._is_excluded(message):
            return
        if not message.strip():
            return
        if not self._should_log(level, message, allow_repeat):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level}] {message}"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def info(self, message, allow_repeat=False):
        self.log(message, level="INFO", allow_repeat=allow_repeat)

    def warn(self, message, allow_repeat=False):
        self.log(message, level="WARN", allow_repeat=allow_repeat)

    def error(self, message, allow_repeat=False):
        self.log(message, level="ERROR", allow_repeat=allow_repeat)

    def debug(self, message, allow_repeat=False):
        self.log(message, level="DEBUG", allow_repeat=allow_repeat)

    def print(self, *args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(a) for a in args)
        if end and end not in ("\n", "\r\n"):
            message = f"{message}{end}"
        self.info(message)


_LOGGER = None


def get_logger(reset=False):
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(reset=reset)
    elif reset:
        _LOGGER.reset()
    return _LOGGER


def install_print_logger(reset=False):
    logger = get_logger(reset=reset)
    import builtins

    builtins.print = logger.print
    return logger

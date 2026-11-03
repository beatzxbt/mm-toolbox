"""File log handler for advanced logging."""

import os
from typing import TextIO

from mm_toolbox.logging._remote import SHUTDOWN_TIMEOUT_S
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler
from mm_toolbox.logging.advanced.pylog import PyLog


class FileLogHandler(BaseLogHandler):
    """Append batches through one file handle, retained until close."""

    def __init__(self, filepath: str, create: bool = False) -> None:
        """Prepare a .txt file, optionally creating it and its parent directories.

        With create=True the file is opened immediately in append mode.
        Otherwise the existing file is opened on the first nonempty push.
        """
        super().__init__()
        self._file: TextIO | None = None
        if not filepath.endswith(".txt"):
            raise ValueError(
                f"Invalid filepath; expected string ending with '.txt' but got "
                f"'{filepath}'"
            )
        self.filepath = filepath
        self.create = create
        if self.create:
            try:
                with self._lifecycle_lock:
                    self._open_file()
            except Exception as exc:
                self._handle_exception(exc, "init")

    def _open_file(self) -> None:
        """Open once while holding the lifecycle lock."""
        if self.create:
            directory = os.path.dirname(self.filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
        elif not os.path.exists(self.filepath):
            raise FileNotFoundError("Target file does not exist and create=False")
        self._file = open(self.filepath, "a", encoding="utf-8")

    def push(self, logs: list[PyLog]) -> None:
        """Append and flush one batch, serializing writes with other pushes/close."""
        if not logs:
            return
        try:
            payload = "\n".join(self.format_log(log) for log in logs) + "\n"
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("handler is closed")
                if self._file is None:
                    self._open_file()
                assert self._file is not None
                self._file.write(payload)
                self._file.flush()
        except Exception as exc:
            self._handle_exception(exc, "push")

    def close(self, timeout_s: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Flush and release the retained file handle once, then close base resources."""
        error: Exception | None = None
        with self._lifecycle_lock:
            self._closed = True
            file, self._file = self._file, None
            if file is not None:
                try:
                    file.close()
                except Exception as exc:
                    error = exc
        super().close(timeout_s=timeout_s)
        if error is not None:
            self._handle_exception(error, "close")

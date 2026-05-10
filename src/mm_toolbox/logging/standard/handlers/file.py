"""File handler for the standard logger.

Appends buffered log messages to a local text file, optionally
creating parent directories and the file itself on first open.
"""

import os

from mm_toolbox.logging.standard.handlers.base import BaseLogHandler


class FileLogHandler(BaseLogHandler):
    """A log handler that appends log messages to a text file."""

    def __init__(self, filepath: str, create: bool = False) -> None:
        """Initialize the FileLogHandler with a target file path.

        Args:
            filepath (str): Path to the text file for appending logs.
                Must end with ".txt".
            create (bool): If True, create the file and parent directories
                if they do not exist. Defaults to False.

        Raises:
            ValueError: If the provided filepath does not end with ".txt".

        """
        super().__init__()

        if not filepath.endswith(".txt"):
            raise ValueError(
                f"Invalid filepath; expected string ending with '.txt' but got "
                f"{filepath}"
            )

        self.filepath = filepath
        self._create = create
        self._file = None

    def open(self) -> None:
        """Open the file for appending. Called by Logger."""
        super().open()

        if self._create:
            directory = os.path.dirname(self.filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            if not os.path.exists(self.filepath):
                with open(self.filepath, "w"):
                    pass

        self._file = open(self.filepath, "a")

    def push(self, buffer: list[str]) -> None:
        """Append buffered messages to the file.

        Args:
            buffer (list[str]): List of formatted log messages.

        """
        if self._file is None:
            return
        self._file.write("\n".join(buffer) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
        super().close()

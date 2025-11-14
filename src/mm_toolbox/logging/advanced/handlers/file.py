"""File log handler for advanced logging."""

import os

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler


class FileLogHandler(BaseLogHandler):
    """A log handler that appends log messages to a text file."""

    def __init__(self, filepath: str, create: bool = False):
        """Initialize the FileLogHandler with a target file path.

        Args:
            filepath (str): Path to the text file for appending logs.
                Must end with ".txt".
            create (bool, optional): If True, create the file if it doesn't
                exist. Defaults to False.

        Raises:
            ValueError: If the provided filepath does not end with ".txt".

        """
        super().__init__()
        if not filepath.endswith(".txt"):
            raise ValueError(
                f"Invalid filepath; expected string ending with '.txt' but got "
                f"'{filepath}'"
            )
        self.filepath = filepath
        self.create = create

        if self.create:
            # Create the file if it doesn't exist, or truncate it if it does
            try:
                directory = os.path.dirname(self.filepath)
                if directory:
                    os.makedirs(directory, exist_ok=True)

                # Create or truncate the file
                with open(self.filepath, "w"):
                    pass
            except Exception as e:
                print(f"Failed to create or truncate file; {e}")

    def push(self, logs):
        try:
            if not os.path.exists(self.filepath):
                if not self.create:
                    print(
                        "Failed to write logs to file; target file does not exist and create=False"
                    )
                    return
                # If create=True and file missing, create parent dirs and file
                directory = os.path.dirname(self.filepath)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with open(self.filepath, "w"):
                    pass
            with open(self.filepath, "a") as file:
                msgs = "\n".join([self.format_log(log) for log in logs]) + "\n"
                file.write(msgs)
                file.flush()

        except Exception as e:
            print(f"Failed to write logs to file; {e}")

from .base import LogHandler

class FileLogHandler(LogHandler):
    """
    A log handler that appends log messages to a text file.
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialize the FileLogHandler with a target file path.

        Args:
            filepath (str): Path to the text file for appending logs. Must end with ".txt".

        Raises:
            ValueError: If the provided filepath does not end with ".txt".
        """
        if not filepath.endswith(".txt"):
            raise ValueError(f"Invalid filepath; expected string ending with '.txt' but got {filepath}")
        self.filepath = filepath

    def push(self, buffer) -> None:
        with open(self.filepath, "a") as file:
            log_msgs = (f"{log.time} - {log.level} - {log.msg}" for log in buffer.data)
            combined_logs = "\n".join(log_msgs) + "\n"
            file.write(combined_logs)
            file.flush()

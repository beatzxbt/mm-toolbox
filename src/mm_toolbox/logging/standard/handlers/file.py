from dataclasses import dataclass
from .base import LogConfig, LogHandler


@dataclass
class FileLogConfig(LogConfig):
    filepath: str = ""

    def validate(self) -> None:
        if not self.filepath or not self.filepath.endswith(".txt"):
            raise ValueError("Missing/Invalid filepath.")
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be greater than 0")
        if self.flush_interval <= 0:
            raise ValueError("Flush interval must be greater than 0")


class FileLogHandler(LogHandler):
    def __init__(self, config: FileLogConfig) -> None:
        config.validate()

        self.log_file = open(config.filepath, "w")

    async def flush(self, buffer) -> None:
        combined_logs = "\n".join(buffer) + "\n"
        self.log_file.write(combined_logs)
        self.log_file.flush()

    async def close(self) -> None:
        self.log_file.close()

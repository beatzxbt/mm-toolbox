"""Configuration classes and enums for standard logging."""

from enum import IntEnum


class LogLevel(IntEnum):
    """Log level enumeration."""

    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


class LoggerConfig:
    """Configuration for standard logger."""

    def __init__(
        self,
        base_level: LogLevel = LogLevel.INFO,
        do_stout: bool = True,
        str_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        flush_interval_s: float = 1.0,
        buffer_size: int = 10000,
    ):
        """Initializes the LoggerConfig.

        Args:
            base_level (LogLevel): The minimum log level that will be logged.
                Defaults to LogLevel.INFO.
            flush_interval_s (float): Maximum time (in seconds) before forcing
                a buffer flush, even if it's not full. Must be > 0. Defaults to 1.0.
            do_stout (bool): If True, logs are also printed to stdout. Defaults to False.
            str_format (str): The format string for log messages.
                Supports %(asctime)s, %(levelname)s, %(name)s, and %(message)s.
                Defaults to "%(asctime)s [%(levelname)s] %(name)s - %(message)s".
            buffer_size (int): The size of the log message buffer. Defaults to 10000.

        Raises:
            ValueError: If flush_interval_s <= 0.
            ValueError: If str_format does not contain '%(message)s' placeholder.

        """
        self.base_level = base_level
        self.do_stout = do_stout

        self.flush_interval_s = flush_interval_s
        if self.flush_interval_s <= 0.0:
            raise ValueError(
                f"Invalid flush interval; expected >0 but got {self.flush_interval_s}"
            )

        self.str_format = str_format
        if "%(message)s" not in self.str_format:
            raise ValueError("Format string must contain '%(message)s' placeholder")

        self.buffer_size = buffer_size
        if self.buffer_size <= 0:
            raise ValueError(
                f"Invalid buffer size; expected >0 but got {self.buffer_size}"
            )

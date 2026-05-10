"""Configuration classes and enums for the standard buffered logger.

Provides ``LogLevel`` severity enumeration and ``LoggerConfig`` for
controlling output formatting, flush thresholds, and atexit behavior.
"""

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
        do_stdout: bool = True,
        str_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        flush_on_size: bool = True,
        flush_size_threshold: int = 1000,
        flush_on_interval: bool = False,
        flush_interval_s: float = 1.0,
        auto_flush_on_exit: bool = True,
    ):
        """Initializes the LoggerConfig.

        Args:
            base_level (LogLevel): The minimum log level that will be logged.
                Defaults to LogLevel.INFO.
            do_stdout (bool): If True, logs are also printed to stdout.
                Defaults to True.
            str_format (str): The format string for log messages.
                Supports %(asctime)s, %(levelname)s, %(name)s, and %(message)s.
                Defaults to "%(asctime)s [%(levelname)s] %(name)s - %(message)s".
            flush_on_size (bool): If True, flush when buffer reaches
                flush_size_threshold. Defaults to True.
            flush_size_threshold (int): Number of messages that triggers an
                immediate flush when flush_on_size is True. Must be > 0.
                Defaults to 1000.
            flush_on_interval (bool): If True, flush after flush_interval_s
                seconds regardless of buffer size. Defaults to False.
            flush_interval_s (float): Maximum time (in seconds) before forcing
                a buffer flush, when flush_on_interval is True. Must be > 0.
                Defaults to 1.0.
            auto_flush_on_exit (bool): If True, registers an atexit hook to
                flush remaining logs on interpreter shutdown. Defaults to True.

        Raises:
            ValueError: If flush_interval_s <= 0 when flush_on_interval is True.
            ValueError: If flush_size_threshold <= 0 when flush_on_size is True.
            ValueError: If str_format does not contain '%(message)s' placeholder.

        """
        self.base_level = base_level
        self.do_stdout = do_stdout

        self.flush_on_size = flush_on_size
        self.flush_size_threshold = flush_size_threshold
        if self.flush_on_size and self.flush_size_threshold <= 0:
            raise ValueError(
                f"Invalid flush size threshold; expected >0 but got "
                f"{self.flush_size_threshold}"
            )

        self.flush_on_interval = flush_on_interval
        self.flush_interval_s = flush_interval_s
        if self.flush_on_interval and self.flush_interval_s <= 0.0:
            raise ValueError(
                f"Invalid flush interval; expected >0 but got {self.flush_interval_s}"
            )

        self.str_format = str_format
        if "%(message)s" not in self.str_format:
            raise ValueError("Format string must contain '%(message)s' placeholder")

        self.auto_flush_on_exit = auto_flush_on_exit

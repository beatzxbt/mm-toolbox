"""Configuration classes and enums for the standard buffered logger.

Provides ``LogLevel`` severity enumeration and ``LoggerConfig`` for
controlling output formatting and flush thresholds.
"""

from enum import IntEnum
from math import isfinite
from typing import Final


class LogLevel(IntEnum):
    """Log level enumeration."""

    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4


TRACE_LEVEL: Final = LogLevel.TRACE.value
DEBUG_LEVEL: Final = LogLevel.DEBUG.value
INFO_LEVEL: Final = LogLevel.INFO.value
WARNING_LEVEL: Final = LogLevel.WARNING.value
ERROR_LEVEL: Final = LogLevel.ERROR.value


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
    ) -> None:
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
                seconds have elapsed when a subsequent log call is processed.
                There is no background interval timer. Defaults to False.
            flush_interval_s (float): Elapsed-time threshold checked on each
                log call when flush_on_interval is True. Must be > 0.
                Defaults to 1.0.

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
        if self.flush_on_interval and (
            not isfinite(self.flush_interval_s) or self.flush_interval_s <= 0.0
        ):
            raise ValueError(
                f"Invalid flush interval; expected >0 but got {self.flush_interval_s}"
            )

        self.str_format = str_format
        if "%(message)s" not in self.str_format:
            raise ValueError("Format string must contain '%(message)s' placeholder")

    @property
    def base_level(self) -> LogLevel:
        return self._base_level

    @base_level.setter
    def base_level(self, level: LogLevel) -> None:
        self.set_base_level(level)

    def set_base_level(self, level: LogLevel) -> None:
        """Change severity and rebuild its integer cache for all sharing loggers."""
        validated = LogLevel(level)
        self._base_level = validated
        self._base_level_value: int = validated.value

"""Python-accessible types for the advanced logging system."""

from enum import IntEnum
from typing import Self

from msgspec import Struct


class PyLogLevel(IntEnum):
    """Python-accessible log level enumeration."""

    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

    def is_lower(self, other: Self) -> bool:
        """Check if this log level is lower than another log level."""
        return self.value < other.value

    def is_higher(self, other: Self) -> bool:
        """Check if this log level is higher than another log level."""
        return self.value > other.value


class PyLog(Struct):
    """Python-accessible log message structure."""

    timestamp_ns: int
    name: bytes
    level: PyLogLevel
    message: bytes

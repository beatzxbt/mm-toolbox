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

    def __iter__(self):
        yield self.timestamp_ns
        yield self.name
        yield self.level
        yield self.message

    def __getitem__(self, index: int):
        if index == 0:
            return self.timestamp_ns
        if index == 1:
            return self.name
        if index == 2:
            return self.level
        if index == 3:
            return self.message
        raise IndexError("PyLog index out of range")

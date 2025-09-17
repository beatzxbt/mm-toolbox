"""Type stubs for pylog module."""

from enum import IntEnum
from typing import Self

import msgspec

class PyLogLevel(IntEnum):
    """Python-accessible log level enumeration."""

    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4

    def is_lower(self, other: Self) -> bool:
        """Check if this log level is lower than another log level."""
        ...

    def is_higher(self, other: Self) -> bool:
        """Check if this log level is higher than another log level."""
        ...

class PyLog(msgspec.Struct):
    """Python-accessible log message structure."""

    timestamp_ns: int
    name: bytes
    level: PyLogLevel
    message: bytes

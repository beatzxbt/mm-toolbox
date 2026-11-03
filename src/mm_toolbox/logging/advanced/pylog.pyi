"""Type stubs for pylog module."""

from collections.abc import Iterator
from enum import IntEnum
from typing import Literal, Self, overload

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
    name: bytes | memoryview
    level: PyLogLevel
    message: bytes | memoryview

    def __init__(
        self,
        timestamp_ns: int,
        name: bytes | memoryview,
        level: PyLogLevel,
        message: bytes | memoryview,
    ) -> None: ...
    def __iter__(self) -> Iterator[int | bytes | memoryview]: ...
    @overload
    def __getitem__(self, index: Literal[0]) -> int: ...
    @overload
    def __getitem__(self, index: Literal[1, 3]) -> bytes | memoryview: ...
    @overload
    def __getitem__(self, index: Literal[2]) -> PyLogLevel: ...
    @overload
    def __getitem__(self, index: int) -> int | bytes | memoryview: ...

def format_log(log: PyLog, str_format: str) -> str: ...

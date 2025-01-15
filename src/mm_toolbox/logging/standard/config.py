from enum import Enum

class LogLevel(Enum):
    """
    Defines the severity levels for log messages.
    """

    TRACE = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    CRITICAL = 6


class LoggerConfig:
    """
    Holds logger configuration parameters, such as base level, 
    buffer capacity, and buffer timeout.
    """

    def __init__(
        self, 
        base_level: LogLevel = LogLevel.INFO,
        stout: bool = True,
        buffer_capacity: int = 10,
        buffer_timeout_s: float = 5.0,
    ): 
        """
        Initializes the LoggerConfig.

        Args:
            base_level (LogLevel): The minimum log level that will be logged.
                Defaults to LogLevel.INFO.
            stout (bool): If True, logs are also printed to stdout. Defaults to True.
            buffer_capacity (int): Maximum number of messages in the buffer
                before forcing a flush. Must be > 1. Defaults to 10.
            buffer_timeout_s (float): Maximum time (in seconds) before forcing
                a buffer flush, even if it's not full. Must be > 0. Defaults to 5.0.

        Raises:
            ValueError: If buffer_capacity <= 1 or buffer_timeout_s <= 0.
        """
        self.base_level = base_level
        self.stout = stout
        self.buffer_capacity = buffer_capacity
        self.buffer_timeout_s = buffer_timeout_s
        
        if self.buffer_capacity <= 1:
            raise ValueError(
                f"Invalid buffer capacity; expected >1 but got {self.buffer_capacity}"
            )
        
        if self.buffer_timeout_s <= 0.0:
            raise ValueError(
                f"Invalid buffer timeout; expected >0 but got {self.buffer_timeout_s}"
            )

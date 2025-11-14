"""Type stubs for master module."""

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler

class MasterLogger:
    """Central aggregator for log messages sent by worker loggers.

    It receives binary messages from workers, decodes them, and forwards
    them to handlers.

    Also can act as a logger itself, but it is not recommended to use it
    for this purpose.
    """

    def __init__(
        self,
        config: LoggerConfig | None = None,
        log_handlers: list[BaseLogHandler] | None = None,
    ) -> None:
        """Initialize a MasterLogger.

        Args:
            config: Configuration for the logger. If None, uses default LoggerConfig.
            log_handlers: List of handlers to forward logs to. If None, uses empty list.
        """
        ...

    def trace(self, msg_str: str | None = None, msg_bytes: bytes = b"") -> None:
        """Send a trace-level log message."""
        ...

    def debug(self, msg_str: str | None = None, msg_bytes: bytes = b"") -> None:
        """Send a debug-level log message."""
        ...

    def info(self, msg_str: str | None = None, msg_bytes: bytes = b"") -> None:
        """Send an info-level log message."""
        ...

    def warning(self, msg_str: str | None = None, msg_bytes: bytes = b"") -> None:
        """Send a warning-level log message."""
        ...

    def error(self, msg_str: str | None = None, msg_bytes: bytes = b"") -> None:
        """Send an error-level log message."""
        ...

    def shutdown(self) -> None:
        """Flush any remaining messages and shuts down the master logger.

        This method stops accepting new messages from worker loggers and
        then stops the connection.

        Warning:
            After calling shutdown(), this logger cannot be used again.
        """
        ...

    def is_running(self) -> bool:
        """Check if the master logger is running."""
        ...

    def get_config(self) -> LoggerConfig:
        """Get the configuration of the master logger."""
        ...

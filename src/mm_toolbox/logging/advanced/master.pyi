"""Type stubs for advanced logging master logger."""

from mm_toolbox.logging.advanced.config import LoggerConfig
from mm_toolbox.logging.advanced.structs import LogLevel

class MasterLogger:
    """The MasterLogger acts as a central aggregator for log messages sent by worker loggers.

    It receives binary, multipart messages, decodes them, and stores them in a buffer until full.
    Once the buffer is full, it flushes them to external handlers.
    """

    def __init__(
        self,
        config: LoggerConfig = ...,
        log_handlers: list = ...,
    ) -> None: ...
    def _process_worker_msg(self, msg: bytes) -> None:
        """Process received multipart messages and decode them into appropriate message structs.

        Args:
            msg (bytes): The raw message bytes to process.

        """
        ...

    def _timed_operations(self) -> None:
        """Background thread that periodically flushes the local log buffer and.

        checks worker heartbeats for being late.
        """
        ...

    def set_log_level(self, level: LogLevel) -> None:
        """Modify the logger's base log level at runtime.

        Args:
            level (LogLevel): The new base log level.

        """
        ...

    def trace(self, msg_str: str = ..., msg_bytes: bytes = ...) -> None:
        """Send a trace-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.

        """
        ...

    def debug(self, msg_str: str = ..., msg_bytes: bytes = ...) -> None:
        """Send a debug-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.

        """
        ...

    def info(self, msg_str: str = ..., msg_bytes: bytes = ...) -> None:
        """Send an info-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.

        """
        ...

    def warning(self, msg_str: str = ..., msg_bytes: bytes = ...) -> None:
        """Send a warning-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.

        """
        ...

    def error(self, msg_str: str = ..., msg_bytes: bytes = ...) -> None:
        """Send an error-level log message.

        Args:
            msg_str (str, optional): The log message text as a string.
            msg_bytes (bytes, optional): The log message text as bytes.

        """
        ...

    def shutdown(self) -> None:
        """Flush any remaining messages and shuts down the master logger.

        This method stops accepting new messages from worker loggers and
        then stops the connection.

        Warning:
            After calling `shutdown()`, this logger cannot be used again.

        """
        ...

    def is_running(self) -> bool:
        """Check if the master logger is running."""
        ...

    def get_config(self) -> LoggerConfig:
        """Get the configuration of the master logger."""
        ...

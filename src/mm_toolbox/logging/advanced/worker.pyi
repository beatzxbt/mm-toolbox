"""Type stubs for worker module."""

from mm_toolbox.logging.advanced.config import LoggerConfig

class WorkerLogger:
    """Lightweight worker logger that sends log messages to the master logger."""

    def __init__(
        self,
        config: LoggerConfig | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a WorkerLogger.

        Args:
            config: Configuration for the logger. If None, uses default LoggerConfig.
            name: Name for the logger. If None, uses f"WORKER{os.getpid()}".
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
        """Shutdown with proper cleanup."""
        ...

    def is_running(self) -> bool:
        """Check if the logger is running."""
        ...

    def get_name(self) -> str:
        """Get the name of the logger."""
        ...

    def get_config(self) -> LoggerConfig:
        """Get the configuration of the logger."""
        ...

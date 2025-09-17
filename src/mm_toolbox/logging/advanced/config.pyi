"""Type stubs for config module."""

from mm_toolbox.logging.advanced.pylog import PyLogLevel

class LoggerConfig:
    """Configuration class for the logging system."""

    base_level: int
    do_stout: bool
    str_format: str
    path: str
    flush_interval_s: float

    def __init__(
        self,
        base_level: PyLogLevel | None = None,
        do_stout: bool = False,
        str_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        path: str = "ipc:///tmp/hft_logger",
        flush_interval_s: float = 1.0,
    ) -> None:
        """Initialize the LoggerConfig with transport, path, and format settings.

        Args:
            base_level: The minimum log level to process. Defaults to PyLogLevel.INFO.
            do_stout: Whether to print to stdout. Defaults to False.
            str_format: The format string for log messages, similar to Python's
                default logger. Supports standard format placeholders like
                %(asctime)s, %(name)s, %(levelname)s, and %(message)s.
                Defaults to '%(asctime)s [%(levelname)s] %(name)s - %(message)s'.
                Must contain at least %(message)s to be valid.
            path: The connection path for the transport protocol. Must follow the format
                required by the selected transport protocol. Defaults to 'ipc:///tmp/hft_logger'.
            flush_interval_s: Timeout in seconds for log messages. Defaults to 1.0.
        """
        ...

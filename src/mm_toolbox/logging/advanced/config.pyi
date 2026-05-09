"""Type stubs for config module."""

from mm_toolbox.logging.advanced.pylog import PyLogLevel

class LoggerConfig:
    """Configuration class for the logging system."""

    base_level: int
    do_stdout: bool
    str_format: str
    path: str
    flush_interval_s: float
    emit_internal: bool
    shm_capacity_bytes: int
    max_batch_messages: int
    max_batch_bytes: int

    def __init__(
        self,
        base_level: PyLogLevel | None = None,
        do_stdout: bool = False,
        str_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        path: str = "/tmp/hft_logger.shm",
        flush_interval_s: float = 1.0,
        emit_internal: bool = False,
        shm_capacity_bytes: int = 67108864,
        max_batch_messages: int = 10000,
        max_batch_bytes: int = 1048576,
    ) -> None:
        """Initialize the LoggerConfig with transport, path, and format settings.

        Args:
            base_level: The minimum log level to process. Defaults to PyLogLevel.INFO.
            do_stdout: Whether to print to stdout. Defaults to False.
            str_format: The format string for log messages, similar to Python's
                default logger. Supports standard format placeholders like
                %(asctime)s, %(name)s, %(levelname)s, and %(message)s.
                Defaults to '%(asctime)s [%(levelname)s] %(name)s - %(message)s'.
                Must contain at least %(message)s to be valid.
            path: The connection path for the transport protocol. Defaults to '/tmp/hft_logger.shm'.
            flush_interval_s: Timeout in seconds for log messages. Defaults to 1.0.
            emit_internal: If True, emit internal startup/shutdown logs. Defaults to False.
            shm_capacity_bytes: Shared memory ring buffer capacity in bytes. Defaults to 64MB.
            max_batch_messages: Max messages before forced flush. Defaults to 10000.
            max_batch_bytes: Max bytes before forced flush. Defaults to 1MB.
        """
        ...

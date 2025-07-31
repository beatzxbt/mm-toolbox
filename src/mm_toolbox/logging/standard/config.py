from enum import StrEnum


class LogLevel(StrEnum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerConfig:
    def __init__(
        self,
        base_level: LogLevel = LogLevel.INFO,
        do_stout: bool = True,
        str_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        flush_interval_s: float = 1.0,
    ):
        """
        Initializes the LoggerConfig.

        Args:
            base_level (LogLevel): The minimum log level that will be logged.
                Defaults to LogLevel.INFO.
            flush_interval_s (float): Maximum time (in seconds) before forcing
                a buffer flush, even if it's not full. Must be > 0. Defaults to 1.0.
            do_stout (bool): If True, logs are also printed to stdout. Defaults to False.
            str_format (str): The format string for log messages.
                Supports %(asctime)s, %(levelname)s, %(name)s, and %(message)s.
                Defaults to "%(asctime)s [%(levelname)s] %(name)s - %(message)s".

        Raises:
            ValueError: If flush_interval_s <= 0.
            ValueError: If str_format does not contain '%(message)s' placeholder.
        """
        self.base_level = base_level
        self.do_stout = do_stout

        self.flush_interval_s = flush_interval_s
        if self.flush_interval_s <= 0.0:
            raise ValueError(
                f"Invalid flush interval; expected >0 but got {self.flush_interval_s}"
            )

        self.str_format = str_format
        if "%(message)s" not in self.str_format:
            raise ValueError("Format string must contain '%(message)s' placeholder")

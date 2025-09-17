from mm_toolbox.logging.advanced.pylog import PyLogLevel
from mm_toolbox.logging.advanced.log cimport CLogLevel


cdef class LoggerConfig:
    def __init__(
        self, 
        object base_level=None,
        bint do_stout=False,
        str str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", 
        str path="ipc:///tmp/hft_logger", 
        double flush_interval_s=1.0,
    ):
        """
        Initialize the LoggerConfig with transport, path, and format settings.
        
        Args:
            base_level (PyLogLevel): The minimum log level to process. Defaults to `PyLogLevel.INFO`.
            do_stout (bool): Whether to print to stdout. Defaults to False.
            str_format (str): The format string for log messages, similar to Python's default logger.
                Supports standard format placeholders like %(asctime)s, %(name)s, %(levelname)s, 
                and %(message)s. Defaults to '%(asctime)s [%(levelname)s] %(name)s - %(message)s'.
                Must contain at least %(message)s to be valid.
            path (str): The connection path for the transport protocol. Must follow the format
                required by the selected transport protocol. Defaults to 'ipc:///tmp/hft_logger'.
            flush_interval_s (float): Timeout in seconds for log messages. Defaults to 1.0.
        """
        if base_level is None:
            base_level = PyLogLevel.INFO
        self.base_level = self.set_base_level_to_clog_level(base_level)
        self.do_stout = do_stout

        self.str_format = str_format
        if "%(message)s" not in str_format:
            raise ValueError("Format string must contain at least the '%(message)s' placeholder")
        
        # Any formatting issues with the path will be thrown within the ZmqConnection constructor 
        # later on in the __init__() file for both Worker/Master loggers, so no need to handle it here. 
        self.path = path

        self.flush_interval_s = flush_interval_s
        if self.flush_interval_s <= 0.0:
            raise ValueError(f"Invalid flush interval; expected a positive number but got '{self.flush_interval_s}'")

    cdef inline CLogLevel set_base_level_to_clog_level(self, object level):
        """Maps a PyLogLevel to a CLogLevel."""
        if level == PyLogLevel.TRACE:
            return CLogLevel.TRACE
        elif level == PyLogLevel.DEBUG:
            return CLogLevel.DEBUG
        elif level == PyLogLevel.INFO:
            return CLogLevel.INFO
        elif level == PyLogLevel.WARNING:
            return CLogLevel.WARNING
        elif level == PyLogLevel.ERROR:
            return CLogLevel.ERROR
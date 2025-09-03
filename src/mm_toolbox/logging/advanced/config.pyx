from mm_toolbox.logging.advanced.structs import LogLevel
from mm_toolbox.logging.advanced.structs cimport CLogLevel

cdef class LoggerConfig:
    def __init__(
        self, 
        LogLevel base_level=LogLevel.INFO,
        bint do_stout=False,
        str str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", 
        str path="ipc:///tmp/hft_logger", 
        double flush_interval_s=1.0,
    ):
        """
        Initialize the LoggerConfig with transport, path, and format settings.
        
        Args:
            base_level (LogLevel): The minimum log level to process. Defaults to `LogLevel.INFO`.
            do_stout (bool): Whether to print to stdout. Defaults to False.
            str_format (str): The format string for log messages, similar to Python's default logger.
                Supports standard format placeholders like %(asctime)s, %(name)s, %(levelname)s, 
                and %(message)s. Defaults to '%(asctime)s [%(levelname)s] %(name)s - %(message)s'.
                Must contain at least %(message)s to be valid.
            path (str): The connection path for the transport protocol. Must follow the format
                required by the selected transport protocol. Defaults to 'ipc:///tmp/hft_logger'.
            flush_interval_s (float): Timeout in seconds for log messages. Defaults to 1.0.
        """
        self.set_base_level(base_level)
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

    cdef inline void set_base_level(self, LogLevel level):
        """
        Maps a LogLevel to a CLogLevel.
        
        Args:
            level (LogLevel): The log level to map.
        """
        if level == LogLevel.TRACE:
            self.base_level = CLogLevel.CTRACE
        elif level == LogLevel.DEBUG:
            self.base_level = CLogLevel.CDEBUG
        elif level == LogLevel.INFO:
            self.base_level = CLogLevel.CINFO
        elif level == LogLevel.WARNING:
            self.base_level = CLogLevel.CWARNING
        elif level == LogLevel.ERROR:
            self.base_level = CLogLevel.CERROR
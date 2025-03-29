from .structs import LogLevelMap
from .structs cimport LogLevel, log_level_to_str

cdef class LoggerConfig:
    def __init__(
        self, 
        Py_ssize_t base_level=LogLevel.INFO,
        bint do_stout=False,
        str str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", 
        str path="ipc:///tmp/hft_logger", 
        double log_timeout_s=2.0,
        Py_ssize_t log_buffer_size=1000,
        double data_timeout_s=5.0,
        Py_ssize_t data_buffer_size=1000,
    ):
        """
        Initialize the LoggerConfig with transport, path, and format settings.
        
        Args:
            base_level (int): The minimum log level to process. Defaults to `LogLevel.INFO`.
            do_stout (bool): Whether to print to stdout. Defaults to False.
            str_format (str): The format string for log messages, similar to Python's default logger.
                Supports standard format placeholders like %(asctime)s, %(name)s, %(levelname)s, 
                and %(message)s. Defaults to '%(asctime)s [%(levelname)s] %(name)s - %(message)s'.
                Must contain at least %(message)s to be valid.
            path (str): The connection path for the transport protocol. Must follow the format
                required by the selected transport protocol. Defaults to 'ipc:///tmp/hft_logger'.
            log_timeout_s (float): Timeout in seconds for log messages. Defaults to 2.0.
            log_buffer_size (int): The maximum number of log messages this buffer can hold. Defaults to 1000.
            data_timeout_s (float): Timeout in seconds for data messages. Defaults to 5.0.
            data_buffer_size (int): The maximum number of data messages this buffer can hold. Defaults to 1000.
        Raises:
            ValueError: If the transport is not one of the supported types, if the path
                does not follow the required format for the selected transport, or if the
                format string does not contain the required %(message)s placeholder.
        """
        # As we're using Cython enums, it doesnt keep the value as an enum type. So we 
        # need to verify the level's validity manually. 
        self.base_level = base_level
        if self.base_level not in LogLevelMap:
            raise ValueError(f"Invalid base level; expected one of '{list(LogLevelMap.values())}' but got '{log_level_to_str(self.base_level)}'")

        self.do_stout = do_stout

        self.str_format = str_format
        # Validate format string
        if "%(message)s" not in str_format:
            raise ValueError("Format string must contain at least the '%(message)s' placeholder")
        
        # Any formatting issues with the path will be thrown within the ZmqConnection constructor 
        # later on in the __init__() file for both Worker/Master loggers, so no need to handle it here. 
        self.path = path

        self.log_timeout_s = log_timeout_s
        if self.log_timeout_s <= 0.0:
            raise ValueError(f"Invalid log timeout; expected a positive number but got '{self.log_timeout_s}'")

        self.data_timeout_s = data_timeout_s
        if self.data_timeout_s <= 0.0:
            raise ValueError(f"Invalid data timeout; expected a positive number but got '{self.data_timeout_s}'")

        self.log_buffer_size = log_buffer_size
        if self.log_buffer_size <= 0:
            raise ValueError(f"Invalid log buffer size; expected a positive number but got '{self.log_buffer_size}'")

        self.data_buffer_size = data_buffer_size
        if self.data_buffer_size <= 0:
            raise ValueError(f"Invalid data buffer size; expected a positive number but got '{self.data_buffer_size}'")
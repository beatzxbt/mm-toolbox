cdef class LoggerConfig:
    cdef:
        public int          base_level
        public bint         do_stout
        public str          str_format
        public str          path
        public double       log_timeout_s
        public Py_ssize_t   log_buffer_size
        public double       data_timeout_s
        public Py_ssize_t   data_buffer_size

    # def __init__(
    #     self, 
    #     int       base_level=LogLevel.INFO,
    #     bint      do_stout=False, 
    #     str       str_format="%(timestamp)s - %(levelname)s - %(message)s", 
    #     str       path="ipc:///logger_queue.ipc", 
    #     double    log_timeout_s=2.0,
    #     Py_ssize_t log_buffer_size=1000,
    #     double    data_timeout_s=5.0,
    #     Py_ssize_t data_buffer_size=1000,
    # )

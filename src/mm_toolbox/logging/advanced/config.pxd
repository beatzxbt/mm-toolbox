from mm_toolbox.logging.advanced.log cimport CLogLevel

cdef class LoggerConfig:
    cdef:
        public CLogLevel    base_level
        public bint         do_stdout
        public str          str_format
        public str          path
        public double       flush_interval_s
        public bint         emit_internal
        public int          ipc_linger_ms

    cdef inline CLogLevel set_base_level_to_clog_level(self, object level)

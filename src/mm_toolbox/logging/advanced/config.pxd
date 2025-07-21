from mm_toolbox.logging.advanced.structs cimport LogLevel, CLogLevel

cdef class LoggerConfig:
    cdef:
        public CLogLevel    base_level
        public bint         do_stout
        public str          str_format
        public str          path
        public double       flush_interval_s

    cdef inline void set_base_level(self, LogLevel level)


cdef class LoggerConfig:
    cdef:
        public str transport
        public str path

    # def __init__(self, str transport="ipc", str path="ipc:///logger_queue.ipc")

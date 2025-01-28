from libc.stdint cimport int64_t

cpdef double time_s()
cpdef double time_ms()
cpdef double time_us()
cpdef int64_t time_ns()
cpdef double iso8601_to_unix(str timestamp)

cdef char* _format_timestamp_ns()
cpdef str time_iso8601()
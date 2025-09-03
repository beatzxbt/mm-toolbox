from libc.stdint cimport int64_t as i64

cpdef i64 time_s()
cpdef i64 time_ms()
cpdef i64 time_us()
cpdef i64 time_ns()
cpdef double iso8601_to_unix(str timestamp)
cpdef str time_iso8601(double timestamp=*)
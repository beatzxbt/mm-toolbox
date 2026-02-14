# distutils: language = c
# distutils: sources = src/mm_toolbox/time/ctime_impl.c
# distutils: include_dirs = src/mm_toolbox/time

import ciso8601
from cpython.unicode cimport PyUnicode_DecodeASCII
from libc.string cimport strlen
from libc.stdint cimport int64_t as i64

cdef extern from "ctime_impl.h":
    i64 c_time_s ()
    i64 c_time_ms ()
    i64 c_time_us ()
    i64 c_time_ns ()
    i64 c_time_monotonic_s ()
    i64 c_time_monotonic_ms ()
    i64 c_time_monotonic_us ()
    i64 c_time_monotonic_ns ()
    char* c_time_iso8601 (double timestamp)
    void c_free_string (char* ptr)

cpdef i64 time_s():
    """Returns the current wall-clock time in seconds."""
    cdef i64 result = c_time_s()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_ms():
    """Returns the current wall-clock time in milliseconds."""
    cdef i64 result = c_time_ms()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_us():
    """Returns the current wall-clock time in microseconds."""
    cdef i64 result = c_time_us()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_ns():
    """Returns the current wall-clock time in nanoseconds."""
    cdef i64 result = c_time_ns()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_monotonic_s():
    """Returns monotonic time in seconds (never decreases, unaffected by clock changes)."""
    cdef i64 result = c_time_monotonic_s()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef i64 time_monotonic_ms():
    """Returns monotonic time in milliseconds (never decreases, unaffected by clock changes)."""
    cdef i64 result = c_time_monotonic_ms()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef i64 time_monotonic_us():
    """Returns monotonic time in microseconds (never decreases, unaffected by clock changes)."""
    cdef i64 result = c_time_monotonic_us()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef i64 time_monotonic_ns():
    """Returns monotonic time in nanoseconds (never decreases, unaffected by clock changes)."""
    cdef i64 result = c_time_monotonic_ns()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef double iso8601_to_unix(str timestamp):
    """Converts an ISO 8601 formatted timestamp to a Unix timestamp."""
    return ciso8601.parse_datetime(timestamp).timestamp()

cpdef str time_iso8601(double timestamp = 0.0):
    """
    Returns an ISO 8601 formatted timestamp.
    
    Args:
        timestamp (float, optional): Unix timestamp to format. If 0.0 (default),
            uses the current time with optimized manual date arithmetic for 
            maximum performance.

    Returns:
        str: The formatted timestamp as 'YYYY-MM-DDTHH:MM:SS.fffZ'.
    """
    cdef char* c_result
    cdef Py_ssize_t c_result_len
    cdef str result

    c_result = c_time_iso8601(timestamp)
    if c_result is NULL:
        raise MemoryError("Failed to allocate memory for timestamp formatting")
    
    try:
        c_result_len = <Py_ssize_t>strlen(c_result)
        result = <str>PyUnicode_DecodeASCII(c_result, c_result_len, NULL)
        return result
    finally:
        c_free_string(c_result)

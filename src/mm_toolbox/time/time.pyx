# distutils: language = c
# distutils: sources = src/mm_toolbox/time/ctime_impl.c
# distutils: include_dirs = src/mm_toolbox/time

import ciso8601
from libc.stdint cimport int64_t as i64
from libc.stddef cimport size_t

cdef extern from "ctime_impl.h":
    i64 c_time_s () nogil
    i64 c_time_ms () nogil
    i64 c_time_us () nogil
    i64 c_time_ns () nogil
    i64 c_time_monotonic_s () nogil
    i64 c_time_monotonic_ms () nogil
    i64 c_time_monotonic_us () nogil
    i64 c_time_monotonic_ns () nogil
    int c_time_iso8601 (double timestamp, char* buf, size_t buf_size) nogil

cpdef i64 time_s():
    """Return the current wall-clock time in seconds.

    Returns:
        Unix timestamp in seconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_s()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_ms():
    """Return the current wall-clock time in milliseconds.

    Returns:
        Unix timestamp in milliseconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_ms()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_us():
    """Return the current wall-clock time in microseconds.

    Returns:
        Unix timestamp in microseconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_us()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_ns():
    """Return the current wall-clock time in nanoseconds.

    Returns:
        Unix timestamp in nanoseconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_ns()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_monotonic_s():
    """Return monotonic time in seconds.

    Monotonic time never decreases and is unaffected by system clock changes.

    Returns:
        Monotonic timestamp in seconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_monotonic_s()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef i64 time_monotonic_ms():
    """Return monotonic time in milliseconds.

    Monotonic time never decreases and is unaffected by system clock changes.

    Returns:
        Monotonic timestamp in milliseconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_monotonic_ms()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef i64 time_monotonic_us():
    """Return monotonic time in microseconds.

    Monotonic time never decreases and is unaffected by system clock changes.

    Returns:
        Monotonic timestamp in microseconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_monotonic_us()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef i64 time_monotonic_ns():
    """Return monotonic time in nanoseconds.

    Monotonic time never decreases and is unaffected by system clock changes.

    Returns:
        Monotonic timestamp in nanoseconds, or raises RuntimeError on failure.
    """
    cdef i64 result
    with nogil:
        result = c_time_monotonic_ns()
    if result == -1:
        raise RuntimeError("Failed to get monotonic time")
    return result

cpdef double iso8601_to_unix(str timestamp):
    """Convert an ISO 8601 formatted timestamp to a Unix timestamp.

    Args:
        timestamp: ISO 8601 formatted string.

    Returns:
        Unix timestamp as a double.
    """
    return ciso8601.parse_datetime(timestamp).timestamp()

cpdef str time_iso8601(double timestamp = 0.0):
    """Return an ISO 8601 formatted timestamp.

    Args:
        timestamp: Unix timestamp to format. If 0.0 (default), uses the
            current time with optimized manual date arithmetic.

    Returns:
        Formatted timestamp as 'YYYY-MM-DDTHH:MM:SS.fffZ'.

    Raises:
        RuntimeError: If formatting fails.
    """
    cdef char buf[64]
    cdef int ret
    
    with nogil:
        ret = c_time_iso8601(timestamp, buf, 64)
    
    if ret != 0:
        raise RuntimeError("Failed to format timestamp")
    
    return buf.decode('ascii')

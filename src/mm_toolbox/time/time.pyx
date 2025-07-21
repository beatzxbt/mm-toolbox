# distutils: language = c
# distutils: sources = src/mm_toolbox/time/ctime_impl.c
# distutils: include_dirs = src/mm_toolbox/time

import ciso8601
from libc.stdint cimport int64_t as i64

# Declare the C functions with aliases to avoid name conflicts
cdef extern from "ctime_impl.h":
    i64 c_time_s ()
    i64 c_time_ms ()
    i64 c_time_us ()
    i64 c_time_ns ()
    char* c_time_iso8601 (double timestamp)
    void c_free_string (char* ptr)

cpdef i64 time_s():
    """
    Returns the current wall-clock time in seconds (float).
    
    Returns:
        float: Current time in seconds since Unix epoch.
        
    Raises:
        RuntimeError: If system clock access fails.
    """
    cdef i64 result = c_time_s()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef double time_ms():
    """
    Returns the current wall-clock time in milliseconds (float).
    
    Returns:
        float: Current time in milliseconds since Unix epoch.
        
    Raises:
        RuntimeError: If system clock access fails.
    """
    cdef double result = c_time_ms()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef double time_us():
    """
    Returns the current wall-clock time in microseconds (float).
    
    Returns:
        float: Current time in microseconds since Unix epoch.
        
    Raises:
        RuntimeError: If system clock access fails.
    """
    cdef double result = c_time_us()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef i64 time_ns():
    """
    Returns the current wall-clock time in nanoseconds (int64).
    
    Returns:
        int64_t: Current time in nanoseconds since Unix epoch.
        
    Raises:
        RuntimeError: If system clock access fails.
    """
    cdef i64 result = c_time_ns()
    if result == -1:
        raise RuntimeError("Failed to get system time")
    return result

cpdef double iso8601_to_unix(str timestamp):
    """
    Converts an ISO 8601 formatted timestamp to a Unix timestamp.

    Args:
        timestamp (str): An ISO 8601 formatted date-time string.

    Returns:
        float: The Unix timestamp corresponding to the provided ISO 8601 date-time.
        
    Raises:
        ValueError: If the timestamp format is invalid.
    """
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
        
    Raises:
        MemoryError: If memory allocation fails.
        RuntimeError: If system clock access fails.
        ValueError: If timestamp is invalid.
    """
    cdef char* c_result
    cdef str result

    c_result = c_time_iso8601(timestamp)
    if c_result == NULL:
        raise MemoryError("Failed to allocate memory for timestamp formatting")
    
    try:
        result = c_result.decode('ascii')
        return result
    finally:
        c_free_string(c_result)
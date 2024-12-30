# Surprisingly, calling the C lib directly gives ~15% speedup across
# the various time functions (compared to importing from time lib directly).
#
# Further optimizations may be found in time_iso8601().
#
# Much credit to Tarasko (author of PicoWs), of where i got the code for 
# clock_gettime(). 

import ciso8601
from time import strftime, gmtime

from libc.stdint cimport int64_t
from libc.errno cimport errno

cdef extern from "<stdlib.h>" nogil:
    enum clockid_t:
        CLOCK_REALTIME
        CLOCK_MONOTONIC
        CLOCK_MONOTONIC_RAW

    cdef struct timespec:
        int64_t tv_sec
        int64_t tv_nsec

    int64_t clock_gettime(clockid_t clock, timespec *ts)

cpdef double time_s():
    """
    Returns the current wall-clock time in seconds (float).
    """
    cdef timespec tspec
    if clock_gettime(CLOCK_REALTIME, &tspec) == -1:
        raise RuntimeError(f"clock_gettime failed: {errno}")
    return <double>tspec.tv_sec + <double>tspec.tv_nsec * 1e-9

cpdef double time_ms():
    """
    Returns the current wall-clock time in milliseconds (float).
    """
    cdef timespec tspec
    if clock_gettime(CLOCK_REALTIME, &tspec) == -1:
        raise RuntimeError(f"clock_gettime failed: {errno}")
    return <double>tspec.tv_sec * 1e3 + <double>tspec.tv_nsec * 1e-6

cpdef double time_us():
    """
    Returns the current wall-clock time in microseconds (float).
    """
    cdef timespec tspec
    if clock_gettime(CLOCK_REALTIME, &tspec) == -1:
        raise RuntimeError(f"clock_gettime failed: {errno}")
    return <double>tspec.tv_sec * 1e6 + <double>tspec.tv_nsec * 1e-3

cpdef int64_t time_ns():
    """
    Returns the current wall-clock time in nanoseconds (float).
    """
    cdef timespec tspec
    if clock_gettime(CLOCK_REALTIME, &tspec) == -1:
        raise RuntimeError(f"clock_gettime failed: {errno}")
    return tspec.tv_sec * 1_000_000_000 + tspec.tv_nsec
 
cpdef str time_iso8601():
    """
    Get the current time in an ISO 8601 formatted timestamp.

    Returns
    -------
    str
        An ISO 8601 formatted date-time string (e.g., "2023-04-04T00:28:50.516Z").
    """
    cdef: 
        int64_t    ms
        str        ms_padded
        timespec   tspec

    if clock_gettime(CLOCK_REALTIME, &tspec) == -1:
        raise RuntimeError(f"clock_gettime failed: {errno}")
    
    ms = tspec.tv_nsec // 1_000_000
    ms_padded = str(ms).zfill(3)

    return f"{strftime('%Y-%m-%dT%H:%M:%S', gmtime())}.{ms_padded}Z"

cpdef double iso8601_to_unix(str timestamp):
    """
    Converts an ISO 8601 formatted timestamp to a Unix timestamp.

    Parameters
    ----------
    timestamp : str
        An ISO 8601 formatted date-time string (e.g., "2023-04-04T00:28:50.516Z").

    Returns
    -------
    float
        The Unix timestamp corresponding to the provided ISO 8601 date-time.
    """
    return ciso8601.parse_datetime(timestamp).timestamp()
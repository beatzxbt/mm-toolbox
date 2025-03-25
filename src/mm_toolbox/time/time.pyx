# Surprisingly, calling the C lib directly gives ~15% speedup across
# the various time functions (compared to importing from time lib directly).
#
# Further optimizations may be found in time_iso8601().
#
# Much credit to Tarasko (author of PicoWs), of where i got the code for 
# clock_gettime(). 

import ciso8601
from datetime import datetime

from libc.stdint cimport int64_t
from libc.errno cimport errno
from libc.string cimport strlen

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

cpdef double iso8601_to_unix(str timestamp):
    """
    Converts an ISO 8601 formatted timestamp to a Unix timestamp.

    Parameters:
        timestamp (str) : An ISO 8601 formatted date-time string.

    Returns:
        float: The Unix timestamp corresponding to the provided ISO 8601 date-time.
    """
    return ciso8601.parse_datetime(timestamp).timestamp()

cpdef str unix_to_iso8601(double timestamp):
    """
    Converts a Unix timestamp to an ISO 8601 formatted timestamp with high precision.

    Parameters:
        timestamp (float): The Unix timestamp to convert. Can be in:
            - seconds (e.g., 1672574400.0)
            - milliseconds (e.g., 1672574400000.0)
            - microseconds (e.g., 1672574400000000.0)
            - nanoseconds (e.g., 1672574400000000000.0)

    Returns:
        str: The ISO 8601 formatted timestamp with full precision.
    """
    cdef:
        double  seconds
        int64_t fractional_part
        str     fractional_str
        str     base_time
        
    if timestamp >= 1e18:  # nanoseconds
        seconds = timestamp / 1e9
        fractional_part = <int64_t>(timestamp % 1e9)
        fractional_str = f"{fractional_part:09d}"
    elif timestamp >= 1e15:  # microseconds
        seconds = timestamp / 1e6
        fractional_part = <int64_t>(timestamp % 1e6)
        fractional_str = f"{fractional_part:06d}"
    elif timestamp >= 1e12:  # milliseconds
        seconds = timestamp / 1e3
        fractional_part = <int64_t>(timestamp % 1e3)
        fractional_str = f"{fractional_part:03d}"
    else:  # seconds
        seconds = timestamp
        fractional_part = <int64_t>((timestamp % 1) * 1e9)
        fractional_str = f"{fractional_part:03d}"

    # Get base time without fractional seconds
    base_time = datetime.fromtimestamp(<int64_t>seconds).isoformat(timespec='seconds')
    
    # Add high precision fractional seconds
    return f"{base_time}.{fractional_str}Z"


# ---------- Time ISO8601 ---------- #


from libc.stdlib cimport malloc, free
from libc.string cimport strlen
from libc.stdio cimport sprintf
from cpython.bytes cimport PyBytes_FromStringAndSize

cdef inline char* _format_timestamp_ns():
    """
    1) Calls time_ns() to get the current time in nanoseconds since 1970-01-01.
    2) Does manual integer arithmetic (Fliegel–Van Flandern style).
    3) Writes the result into a C buffer 'YYYY-MM-DDTHH:MM:SS.fffZ'.

    Returns a pointer to the newly allocated buffer, or NULL if malloc fails.
    The caller must free it.
    """
    # Allocate space for the final ASCII string + null terminator
    cdef char* out_buf = <char*> malloc(40)
    if out_buf == NULL:
        return NULL

    # 1) Get current time in ns from your function
    cdef int64_t nanoseconds = time_ns()

    # 2) Convert nanoseconds to days + remainder
    cdef int64_t NS_PER_DAY = 86400000000000
    cdef long long days_since_epoch = nanoseconds // NS_PER_DAY
    cdef long long remainder_ns = nanoseconds % NS_PER_DAY

    # 3) Do the manual date arithmetic
    cdef long long z = days_since_epoch + 720198 
    cdef long long era
    if z >= 0:
        era = z // 146097
    else:
        era = (z - 146096) // 146097

    cdef long long doe = z - era * 146097
    cdef long long yoe = (doe - (doe // 1460) + (doe // 36524) - (doe // 146096)) // 365
    cdef long long year = yoe + era * 400 - 2  # Added -2 to fix the year error
    cdef long long t1 = (365*yoe + (yoe // 4) - (yoe // 100) + (yoe // 400))
    cdef long long doy = doe - t1
    cdef long long mp = (5*doy + 2) // 153
    cdef long long day = doy - (153 * mp + 2) // 5 + 1
    cdef long long month = mp + 3 if mp < 10 else mp - 9

    if month <= 2:
        year -= 1

    # 4) leftover ns => hour, min, sec, us
    cdef long long ns_in_hour = 3600000000000
    cdef long long ns_in_min  = 60000000000
    cdef long long ns_in_sec  = 1000000000

    cdef long long h = remainder_ns // ns_in_hour
    remainder_ns %= ns_in_hour
    cdef long long M = remainder_ns // ns_in_min
    remainder_ns %= ns_in_min
    cdef long long s = remainder_ns // ns_in_sec
    remainder_ns %= ns_in_sec
    cdef long long ms = remainder_ns // 1_000_000

    # 5) Format to "YYYY-MM-DDTHH:MM:SS.fffZ"
    sprintf(
        out_buf,
        "%04lld-%02lld-%02lldT%02lld:%02lld:%02lld.%03lldZ",
        year, month, day, h, M, s, ms
    )

    return out_buf

cpdef str time_iso8601():
    """
    Returns the current UTC time as 'YYYY-MM-DDTHH:MM:SS.fffZ'.

    Returns:
        str: The formatted date/time.
    """
    cdef char* buf_ptr = _format_timestamp_ns()
    if buf_ptr == NULL:
        raise MemoryError("Failed to allocate timestamp buffer")

    # Convert the C-string to a Python bytes, then decode to str
    cdef int clen = 0
    clen = <int>strlen(buf_ptr) 
    cdef bytes py_bytes = PyBytes_FromStringAndSize(buf_ptr, clen)

    free(buf_ptr)

    # bytes -> str
    return py_bytes.decode("ascii")
import ciso8601
from time import (
    strftime,
    gmtime,
    time_ns as time_nano, 
    time as time_sec,
)

def time_s() -> float:
    """
    Get the current time in seconds since the epoch.

    Returns
    -------
    float
        The current time in seconds.
    """
    return time_sec()

def time_ms() -> float:
    """
    Get the current time in milliseconds since the epoch.

    Returns
    -------
    float
        The current time in milliseconds.
    """
    return time_sec() * 1_000.0

def time_us() -> float:
    """
    Get the current time in microseconds since the epoch.

    Returns
    -------
    float
        The current time in milliseconds.
    """
    return time_sec() * 1_000_000.0

def time_ns() -> int:
    """
    Get the current time in nanoseconds since the epoch.

    Returns
    -------
    int
        The current time in nanoseconds.
    """
    return time_nano()

def time_iso8601() -> str:
    """
    Get the current time in an ISO 8601 formatted timestamp.

    Returns
    -------
    str
        An ISO 8601 formatted date-time string (e.g., "2023-04-04T00:28:50.516Z").
    """
    millis = str((time_nano() % 1_000_000_000) // 1_000_000).zfill(3)
    return f"{strftime('%Y-%m-%dT%H:%M:%S', gmtime())}.{millis}Z"

def iso8601_to_unix(timestamp: str) -> float:
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
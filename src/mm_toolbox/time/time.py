import ciso8601
from time import (
    strftime,
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

def time_ns() -> float:
    """
    Get the current time in milliseconds since the epoch.

    Returns
    -------
    float
        The current time in milliseconds.
    """
    return time_nano()

def datetime_now() -> str:
    """
    Get the current time in the format 'YYYY-MM-DD HH:MM:SS.mmm'.

    Returns
    -------
    str
        The current time string.
    """
    return strftime("%Y-%m-%d %H:%M:%S") + f".{time_ms() % 1000.0}"

def iso8601_to_unix(timestamp: str) -> int:
    """
    Converts an ISO 8601 formatted timestamp to a Unix timestamp.

    This function parses an ISO 8601 date-time string and converts it 
    into a Unix timestamp, which represents the number of seconds 
    that have elapsed since January 1, 1970 (midnight UTC/GMT).

    Parameters
    ----------
    timestamp : str
        An ISO 8601 formatted date-time string (e.g., "2023-04-04T00:28:50.516Z").

    Returns
    -------
    int
        The Unix timestamp corresponding to the provided ISO 8601 date-time.
    
    Example
    -------
    >>> iso8601_to_unix("2023-04-04T00:28:50.516Z")
    1680569330
    """
    return int(ciso8601.parse_datetime(timestamp).timestamp())
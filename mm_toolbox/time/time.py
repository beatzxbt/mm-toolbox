from time import time_ns, strftime

def datetime() -> str:
    """ Millisecond precision datetime, suitable for logging/charting """
    return strftime("%Y-%m-%d %H:%M:%S") + f".{(time_ns()//1000000) % 1000:03d}"

def time_ms() -> int:
    """ Marginally faster than time() * 1_000 """
    return time_ns()//1_000_000

def time_us() -> int:
    """ Marginally faster than time() * 1_000_000 """
    return time_ns()//1_000
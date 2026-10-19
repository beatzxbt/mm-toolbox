"""Native test helpers for supplying deterministic TEMA timestamps."""

from libc.stdint cimport int64_t as i64

from mm_toolbox.moving_average.tema cimport TimeExponentialMovingAverage


def set_time_ns(TimeExponentialMovingAverage tema, i64 time_ns):
    """Set the last-update timestamp for deterministic native tests."""
    tema._time_ns = time_ns


def last_time_ns(TimeExponentialMovingAverage tema):
    """Return the last-update timestamp for deterministic native tests."""
    return tema._time_ns


def next_at(TimeExponentialMovingAverage tema, double value, i64 time_ns):
    """Preview a sample at an explicit monotonic timestamp."""
    return tema._next_at(value, time_ns)


def update_at(TimeExponentialMovingAverage tema, double value, i64 time_ns):
    """Apply a sample at an explicit monotonic timestamp."""
    return tema._update_at(value, time_ns)

"""Time-based exponential moving average (TEMA) implementation.

Uses elapsed time since the last update to compute the smoothing weight,
rather than a fixed window size.
"""

cimport numpy as cnp
from libc.math cimport expm1, isfinite, log
from libc.stdint cimport int64_t as i64

from mm_toolbox.time.time cimport time_monotonic_ns

from mm_toolbox.moving_average.base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    """Time-based exponential moving average.

    The smoothing weight depends on the time elapsed since the last update
    and a configurable half-life.

    Attributes:
        _time_ns (i64): Monotonic timestamp of the last update (nanoseconds).
        _lam (double): Decay constant (ln(2) / half_life_s).
    """

    def __init__(self, int window=2, bint is_fast=False, double half_life_s=1.0):
        """Initialize the time-based EMA.

        Args:
            window (int): Window size (mainly for ring buffer when not fast).
            is_fast (bool): If True, skip storing historical values.
            half_life_s (double): Time in seconds for the weight to halve.

        Raises:
            ValueError: If half_life_s is not positive.
        """
        super().__init__(
            window=window, 
            is_fast=is_fast,
        )

        if not isfinite(half_life_s) or half_life_s <= 0.0:
            raise ValueError("Half life must be positive and finite.")

        self._lam = log(2.0) / half_life_s
        if not isfinite(self._lam):
            raise ValueError("Half life is too small to represent safely.")
        self._time_ns = time_monotonic_ns()

    cdef inline double _alpha_at(self, i64 time_now_ns) noexcept:
        """Calculate smoothing weight for a monotonic timestamp."""
        cdef i64 elapsed_ns = time_now_ns - self._time_ns
        if elapsed_ns <= 0:
            return 0.0
        return -expm1(-self._lam * (<double>elapsed_ns * 1e-9))

    cdef double _next_at(self, double new_val, i64 time_now_ns) noexcept:
        """Calculate a preview at a supplied monotonic timestamp."""
        if not self._is_warm:
            return new_val
        cdef double alpha = self._alpha_at(time_now_ns)
        return alpha * new_val + (1.0 - alpha) * self._value

    cdef double _update_at(self, double new_val, i64 time_now_ns):
        """Apply a sample at a supplied monotonic timestamp."""
        cdef double alpha
        if not self._is_warm:
            self._time_ns = time_now_ns
            self._value = new_val
            self._is_warm = True
            return self._value

        alpha = self._alpha_at(time_now_ns)
        self._time_ns = time_now_ns
        self._value = alpha * new_val + (1.0 - alpha) * self._value
        self.push_to_ringbuffer()
        return self._value

    cpdef double initialize(self, cnp.ndarray values):
        """Warm the TEMA from an array of historical values.

        Historical values have no timestamps, so all warm-up samples share
        one clock origin. The first value seeds the baseline; elapsed-time
        decay begins with subsequent live updates.

        Args:
            values (cnp.ndarray): Array of initial values (length > 1).

        Returns:
            double: The final TEMA value after ingesting all values.

        Raises:
            ValueError: If the input array is too short.
        """
        cdef:
            int i, n = values.shape[0]
            double _temp_var

        if n <= 1:
            raise ValueError(
                f"Input array too short; expected >1 but got {n}"
            )

        self._time_ns = time_monotonic_ns()
        self._values.clear()
        self._value = values[0]
        self.push_to_ringbuffer()
        self._is_warm = True

        for i in range(1, n):
            _temp_var = self._update_at(values[i], self._time_ns)

        return self._value

    cpdef double next(self, double new_val):
        """Calculate the next TEMA value without updating state.

        Args:
            new_val (double): New input value to evaluate.

        Returns:
            double: The projected TEMA value.
        """
        return self._next_at(new_val, time_monotonic_ns())

    cpdef double update(self, double new_val):
        """Update the TEMA with a new value.

        Args:
            new_val (double): New input value to ingest.

        Returns:
            double: The updated TEMA value.
        """
        return self._update_at(new_val, time_monotonic_ns())

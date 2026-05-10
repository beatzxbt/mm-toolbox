"""Time-based exponential moving average (TEMA) implementation.

Uses elapsed time since the last update to compute the smoothing weight,
rather than a fixed window size.
"""

cimport numpy as cnp
from libc.math cimport log, exp

from mm_toolbox.time.time cimport time_s

from mm_toolbox.moving_average.base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    """Time-based exponential moving average.

    The smoothing weight depends on the time elapsed since the last update
    and a configurable half-life.

    Attributes:
        _time_s (double): Timestamp of the last update (seconds).
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

        if half_life_s <= 0.0:
            raise ValueError("Half life must be positive.")

        self._time_s = time_s()
        self._lam = log(2.0) / half_life_s

    cpdef double initialize(self, cnp.ndarray values):
        """Warm the TEMA from an array of historical values.

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

        self._values.clear()
        self._value = values[0]
        self.push_to_ringbuffer()
        self._is_warm = True

        for i in range(1, n):
            _temp_var = self.update(values[i])

        return self._value

    cpdef double next(self, double new_val):
        """Calculate the next TEMA value without updating state.

        Args:
            new_val (double): New input value to evaluate.

        Returns:
            double: The projected TEMA value.
        """
        if not self._is_warm:
            return new_val

        cdef:
            double time_now = time_s()
            double minus_dt = self._time_s - time_now
            double alpha = 1.0 - exp(self._lam * minus_dt)

        return alpha * new_val + (1.0 - alpha) * self._value

    cpdef double update(self, double new_val):
        """Update the TEMA with a new value.

        Args:
            new_val (double): New input value to ingest.

        Returns:
            double: The updated TEMA value.
        """
        if not self._is_warm:
            self._time_s = time_s()
            self._value = new_val
            self._is_warm = True
            return self._value

        cdef:
            double time_now = time_s()
            double minus_dt = self._time_s - time_now
            double alpha = 1.0 - exp(self._lam * minus_dt)

        self._time_s = time_now
        self._value = alpha * new_val + (1.0 - alpha) * self._value
        self.push_to_ringbuffer()

        return self._value

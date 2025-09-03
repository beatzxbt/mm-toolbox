cimport numpy as cnp
from libc.math cimport log, exp

from mm_toolbox.time.time cimport time_s

from mm_toolbox.moving_average.base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    """
    Uses time elapsed since last update to calculate weight of the new value.
    """

    def __init__(self, int window=2, bint is_fast=False, double half_life_s=1.0):
        """
        Window is not a required parameter if is_fast=True, its default is only 
        used to fulfil base class requirements. Though if is_fast=False, it will
        set the length of the ringbuffer. Be weary!
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
        cdef:
            int i, n = values.shape[0]
            double _temp_var

        if n <= 1:
            raise ValueError(
                f"Input array too short; expected >1 but got {n}"
            )

        self._values.fast_reset()
        self._value = values[0]
        self.push_to_ringbuffer()
        self._is_warm = True

        for i in range(1, n):
            _temp_var = self.update(values[i])

        return self._value

    cpdef double next(self, double new_val):
        if not self._is_warm:
            self._time_s = time_s()
            self._value = new_val
            self._is_warm = True
            return self._value

        cdef:
            double time_now = time_s()
            double minus_dt = self._time_s - time_now
            double alpha = 1.0 - exp(self._lam * minus_dt)

        return alpha * new_val + (1.0 - alpha) * self._value

    cpdef double update(self, double new_val):
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

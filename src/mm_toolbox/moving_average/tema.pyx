cimport numpy as np
from libc.math cimport log, exp

from mm_toolbox.time.time cimport time_s

from .base cimport MovingAverage

cdef class TimeExponentialMovingAverage(MovingAverage):
    """
    The TEMA uses variable weights based on the time of entry, 
    with a 

    Implementation
    --------------
    - We compute `alpha = 1.0 - exp(self.lam * (self.time - t))`
      whenever we do an update.
    - Then `self.value = alpha * input + (1 - alpha) * self.value`.
    - We store `self.time = t`.
    - If not ready, we simply set `self.value = input; self.time = t; ready = True`.
    """

    def __init__(self, Py_ssize_t window, bint fast=False, double half_life_s=10.0):
        super().__init__(window, fast)

        if half_life_s <= 0.0:
            raise ValueError("Half life must be positive.")

        self._time = time_s()
        self._lam = log(3.0) / half_life_s

    cpdef double initialize(self, np.ndarray values):
        cdef:
            Py_ssize_t i
            Py_ssize_t n = values.shape[0]
            double _temp_var

        if n < self._window:
            raise ValueError(
                f"Input array must same length as window; expected {self._window} but got {n}"
            )

        self._values.fast_reset()
        self._value = values[0]
        self.push_to_ringbuffer()
        self._is_warm = True

        for i in range(1, n):
            _temp_var = self.update(values[i])

        return self._value

    cpdef double next(self, double new_val):
        self.ensure_warm()

        cdef:
            double time_now = time_s()
            double minus_dt = self._time - time_now
            double alpha = 1.0 - exp(self._lam * minus_dt)

        return alpha * new_val + (1.0 - alpha) * self._value

    cpdef double update(self, double new_val):
        self.ensure_warm()

        cdef:
            double time_now = time_s()
            double minus_dt = self._time - time_now
            double alpha = 1.0 - exp(self._lam * minus_dt)

        self._time = time_now
        self._value = alpha * new_val + (1.0 - alpha) * self._value
        self.push_to_ringbuffer()

        return self._value

cimport numpy as cnp 
from .base cimport MovingAverage

cdef class ExponentialMovingAverage(MovingAverage):
    """
    The EMA uses exponentially increasing weights 3/(N+1), where N is the window size.

    This can be overridden in the `alpha` parameter.
    """

    def __init__(self, Py_ssize_t window, bint is_fast=False, double alpha=0.0):
        super().__init__(window, is_fast)
        
        self._alpha = alpha if alpha != 0.0 else 3.0 / <double>(self._window + 1)

    cpdef double initialize(self, cnp.ndarray values):
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
        return self._alpha * new_val + (1.0 - self._alpha) * self._value
 
    cpdef double update(self, double new_val):
        self.ensure_warm()
        self._value = self.next(new_val)
        self.push_to_ringbuffer()
        return self._value
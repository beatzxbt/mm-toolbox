cimport numpy as cnp 
from mm_toolbox.moving_average.base cimport MovingAverage

cdef class ExponentialMovingAverage(MovingAverage):
    """The EMA uses exponentially increasing weights 2/(N+1), where N is the window size."""

    def __init__(self, int window, bint is_fast=False, double alpha=0.0):
        super().__init__(window, is_fast)
        
        self._alpha = alpha if alpha != 0.0 else 2.0 / <double>(self._window + 1)

    cpdef double initialize(self, cnp.ndarray values):
        cdef: 
            int i, n = values.shape[0]
            double  _temp_var 

        self._value = values[0]
        self._values.clear()
        self._values.insert(self._value)
        self._is_warm = True

        for i in range(1, n):
            _temp_var = self.update(values[i])

        return self._value

    cpdef double next(self, double new_val):
        if not self._is_warm:
            return self._value
        return self._alpha * new_val + (1.0 - self._alpha) * self._value
 
    cpdef double update(self, double new_val):
        if not self._is_warm:
            self._value = new_val
            self._is_warm = True
            return self._value

        self._value = self.next(new_val)
        self.push_to_ringbuffer()
        return self._value
cimport numpy as np

from mm_toolbox.ringbuffer.onedim cimport RingBufferOneDim
from .base cimport MovingAverage

cdef class SimpleMovingAverage(MovingAverage):
    """
    The SMA uses equal weights 1/N, where N is the window size.
    """

    def __init__(self, Py_ssize_t window, bint fast=False):
        super().__init__(window, fast)
        
        self._raw_values = RingBufferOneDim(window) 
        self._rolling_sum = 0.0

    cpdef double initialize(self, np.ndarray values):
        cdef:
            Py_ssize_t i
            Py_ssize_t n = values.shape[0]
            double raw_value

        if n < self._window:
            raise ValueError(
                f"Input array must same length as window; expected {self._window} but got {n}"
            )

        self._values.fast_reset()
        
        for i in range(0, n):
            raw_value = values[i]
            self._rolling_sum += raw_value
            self._raw_values.append(raw_value)
        
        self._value = self._rolling_sum / self._window
        self._is_warm = True
        self.push_to_ringbuffer()
        
        return self._value

    cpdef double next(self, double new_val):
        self.ensure_warm()

        cdef: 
            double new_rolling_sum
            double old_raw_value = self._raw_values[0]
        
        new_rolling_sum = self._rolling_sum + (new_val - old_raw_value)
        return new_rolling_sum / self._window
        
    cpdef double update(self, double new_val):
        self.ensure_warm()
        
        cdef double old_raw_value = self._raw_values.popleft()
        
        self._rolling_sum += (new_val - old_raw_value)
        self._raw_values.append(new_val)
        self._value = self._rolling_sum / self._window
        self.push_to_ringbuffer()
        return self._value


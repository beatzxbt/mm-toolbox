"""Exponential moving average (EMA) implementation.

Uses exponentially increasing weights 2/(N+1), where N is the window size.
"""

cimport numpy as cnp 
from mm_toolbox.moving_average.base cimport MovingAverage

cdef class ExponentialMovingAverage(MovingAverage):
    """Exponential moving average with configurable smoothing factor.

    Attributes:
        _alpha (double): Smoothing factor (default 2/(window+1)).
    """

    def __init__(self, int window, bint is_fast=False, double alpha=0.0):
        """Initialize the EMA.

        Args:
            window (int): Number of samples in the moving average window.
            is_fast (bool): If True, skip storing historical values.
            alpha (double): Custom smoothing factor; 0.0 selects the default.
        """
        super().__init__(window, is_fast)
        
        self._alpha = alpha if alpha != 0.0 else 2.0 / <double>(self._window + 1)

    cpdef double initialize(self, cnp.ndarray values):
        """Warm the EMA from an array of historical values.

        Args:
            values (cnp.ndarray): Initial values for the moving average window.

        Returns:
            double: The final EMA value after ingesting all values.
        """
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
        """Calculate the next EMA value without updating state.

        Args:
            new_val (double): New input value to evaluate.

        Returns:
            double: The projected EMA value.
        """
        if not self._is_warm:
            return self._value
        return self._alpha * new_val + (1.0 - self._alpha) * self._value
 
    cpdef double update(self, double new_val):
        """Update the EMA with a new value.

        Args:
            new_val (double): New input value to ingest.

        Returns:
            double: The updated EMA value.
        """
        if not self._is_warm:
            self._value = new_val
            self._is_warm = True
            return self._value

        self._value = self.next(new_val)
        self.push_to_ringbuffer()
        return self._value
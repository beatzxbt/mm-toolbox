"""Exponential moving average (EMA) implementation.

Uses exponentially increasing weights 2/(N+1), where N is the window size.
"""

cimport numpy as cnp 
from libc.math cimport isfinite
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
            alpha (double): Finite smoothing factor in (0, 1]; 0 selects the default.
        """
        if not isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be 0 for the default, or finite and within (0, 1]")
        super().__init__(window, is_fast)
        
        self._alpha = alpha if alpha != 0.0 else 2.0 / (self._window + 1.0)

    cpdef double initialize(self, cnp.ndarray values):
        """Warm the EMA from an array of historical values.

        Args:
            values (cnp.ndarray): Nonempty one-dimensional finite float64 array.

        Returns:
            double: The final EMA value after ingesting all values.
        """
        cdef: 
            const double[:] values_view = self.validate_values(values)
            Py_ssize_t i, n = values_view.shape[0]

        self._value = values_view[0]
        self.reset_history()
        self.push_to_ringbuffer()
        self._is_warm = True

        for i in range(1, n):
            self.update(values_view[i])

        return self._value

    cpdef double next(self, double new_val):
        """Calculate the next EMA value without updating state.

        Args:
            new_val (double): New input value to evaluate.

        Returns:
            double: The projected EMA value.
        """
        self.validate_value(new_val)
        if not self._is_warm:
            return new_val
        return self._alpha * new_val + (1.0 - self._alpha) * self._value
 
    cpdef double update(self, double new_val):
        """Update the EMA with a new value.

        Args:
            new_val (double): New input value to ingest.

        Returns:
            double: The updated EMA value.
        """
        if not self._is_warm:
            self.validate_value(new_val)
            self._value = new_val
            self._is_warm = True
        else:
            self._value = self.next(new_val)
        self.push_to_ringbuffer()
        return self._value

import numpy as np
from numba.types import bool_, uint32, float64
from numba.experimental import jitclass
from typing import Optional

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat

@jitclass
class WeightedMovingAverage:
    """
    Weighted Moving Average (WMA) with optional RingBuffer to store history.
    """
    window: uint32
    fast: bool_
    value: float64

    _values: RingBufferSingleDimFloat.class_type.instance_type
    _input_values: RingBufferSingleDimFloat.class_type.instance_type
    _weights: float64[:]
    _weight_sum: float64

    def __init__(self, window: int, fast: Optional[bool] = None):
        self.window = window
        self.fast = fast if fast is not None else True
        self.value = 0.0

        self._values = RingBufferSingleDimFloat(self.window)
        self._input_values = RingBufferSingleDimFloat(self.window)
        self._weights = np.arange(self.window, dtype=float64) + 1.0
        self._weight_sum = self._weights.sum()

    def initialize(self, arr_in: np.ndarray) -> None:
        """
        Initializes the WMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : np.ndarray
            The initial series of data points to feed into the WMA calculator.
        """
        assert arr_in.ndim == 1 and arr_in.size >= self.window, (
            f"Input array must be 1D and at least of size {self.window}, "
            f"but got size {arr_in.size}"
        )

        self.value = 0.0
        self._values.reset()
        self._input_values.reset()
        
        for value in arr_in:
            self.update(value)

    def update(self, new_val: float) -> None:
        """
        Updates the WMA calculator with a new data point.

        Parameters:
        -----------
        new_val : float
            The new data point to include in the WMA calculation.
        """
        self._input_values.append(new_val)
        weighted_inputs = self._input_values.as_array() * self._weights[-len(self._input_values):]
        self.value = weighted_inputs.sum() / self._weight_sum
        
        if not self.fast:
            self._values.append(self.value)

    @property
    def values(self) -> np.ndarray:
        return self._values.as_array()
    
    def __eq__(self, wma: "WeightedMovingAverage") -> bool:
        assert isinstance(wma, WeightedMovingAverage)
        return np.all(wma.values == self.values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

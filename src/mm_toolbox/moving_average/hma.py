import numpy as np
# Comment out jitclass for debugging
# from numba.experimental import jitclass
from numba.types import bool_, uint32, float64
from typing import Optional

from .wma import WeightedMovingAverage as WMA
from mm_toolbox.ringbuffer import RingBufferSingleDimFloat

# Comment out jitclass for debugging
# @jitclass
class HullMovingAverage:
    window: uint32
    fast: bool_
    value: float64

    _short_wma: WMA
    _long_wma: WMA
    _smooth_wma: WMA
    _values: RingBufferSingleDimFloat

    def __init__(self, window: int, fast: Optional[bool] = None):
        self.window = window
        self.fast = fast if fast is not None else True
        self.value = 0.0

        self._values = RingBufferSingleDimFloat(window)
        self._short_wma = WMA(int(window / 2), False)
        self._long_wma = WMA(window, False)
        self._smooth_wma = WMA(int(np.sqrt(window)), True)

    def initialize(self, arr_in: np.ndarray[float]) -> None:
        """
        Initializes the HMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : np.ndarray[float]
            The initial series of data points to feed into the HMA calculator.
        """
        assert arr_in.ndim == 1 and arr_in.size >= self.window, (
            f"Input array must be 1D and at least of size {self.window}, "
            f"but got size {arr_in.size}"
        )

        self.value = 0.0
        self._values.reset()
        self._short_wma._values.reset()
        self._long_wma._values.reset()
        self._smooth_wma._values.reset()

        self._short_wma.initialize(arr_in)
        self._long_wma.initialize(arr_in)

        # TODO: Needs optimization to make short/long EMA fast=True.
        short_wma_values = self._short_wma.values
        long_wma_values = self._long_wma.values
        min_length = min(len(short_wma_values), len(long_wma_values))
        diff_series = (short_wma_values[:min_length] * 2.0) - long_wma_values[:min_length]

        self._smooth_wma.initialize(diff_series)
        self.value = self._smooth_wma.value

        if not self.fast:
            self._values.append(self.value)

    def update(self, value: float) -> None:
        """
        Updates the HMA calculator with a new data point.

        Parameters:
        -----------
        value : float
            The new data point to include in the HMA calculation.
        """
        self._short_wma.update(value)
        self._long_wma.update(value)

        diff = (self._short_wma.value * 2.0) - self._long_wma.value
        self._smooth_wma.update(diff)
        self.value = self._smooth_wma.value

        if not self.fast:
            self._values.append(self.value)

    @property
    def values(self) -> np.ndarray:
        return self._values.as_array()

    def __eq__(self, hma: "HullMovingAverage") -> bool:
        assert isinstance(hma, HullMovingAverage)
        return np.all(hma.values == self.values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

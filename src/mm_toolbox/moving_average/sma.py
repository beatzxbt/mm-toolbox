import numpy as np
from numba.types import bool_, uint32, float64
from numba.experimental import jitclass
from typing import Optional

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat


@jitclass
class SimpleMovingAverage:
    """
    Simple Moving Average (SMA) with optional RingBuffer to store history.
    """

    window: uint32
    fast: bool_
    value: float64

    _values: RingBufferSingleDimFloat.class_type.instance_type
    _weighted_values: RingBufferSingleDimFloat.class_type.instance_type
    _weight: float

    def __init__(self, window: int, fast: Optional[bool] = None):
        self.window = window
        self.fast = fast if fast is not None else True
        self.value = 0.0

        self._values = RingBufferSingleDimFloat(self.window)
        self._weighted_values = RingBufferSingleDimFloat(self.window)
        self._weight = 1.0 / window

    def initialize(self, arr_in: np.ndarray[float]) -> None:
        """
        Initializes the SMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : Iterable[float]
            The initial series of data points to feed into the SMA calculator.
        """
        assert arr_in.ndim == 1 and arr_in.size >= self.window, (
            f"Input array must be 1D and at least of size {self.window}, "
            f"but got size {arr_in.size}"
        )

        self.value = 0.0
        self._values.reset()
        self._weighted_values.reset()

        for value in arr_in:
            self.update(value)

    def update(self, new_val: float) -> None:
        """
        Updates the SMA calculator with a new data point.

        Parameters:
        -----------
        new_val : float
            The new data point to include in the SMA calculation.
        """
        if self._weighted_values.is_full:
            self.value -= self._weighted_values.popleft()

        weighted_update = new_val * self._weight
        self._weighted_values.append(weighted_update)
        self.value += weighted_update

        if not self.fast:
            self._values.append(self.value)

    @property
    def values(self) -> np.ndarray:
        return self._values.as_array()

    def __eq__(self, sma: "SimpleMovingAverage") -> bool:
        assert isinstance(sma, SimpleMovingAverage)
        return np.all(sma.values == self.values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

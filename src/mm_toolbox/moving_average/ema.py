import numpy as np
from numba.types import bool_, uint32, float64
from numba.experimental import jitclass
from typing import Optional

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat


@jitclass
class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) with optional RingBuffer to store history.
    """

    window: uint32
    alpha: float64
    fast: bool_
    value: float64
    
    _values: RingBufferSingleDimFloat.class_type.instance_type

    def __init__(
        self, window: int, alpha: Optional[float] = None, fast: Optional[bool] = None
    ):
        self.window = window
        self.alpha = alpha if alpha is not None else 3.0 / (self.window + 1)
        self.fast = fast if fast is not None else True
        self.value = 0.0
        
        self._values = RingBufferSingleDimFloat(self.window)

    def initialize(self, arr_in: np.ndarray[float]) -> None:
        """
        Initializes the EMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : Iterable[float]
            The initial series of data points to feed into the EMA calculator.
        """
        assert arr_in.ndim == 1 and arr_in.size >= self.window, (
            f"Input array must be 1D and at least of size {self.window}, "
            f"but got size {arr_in.size}"
        )

        self._values.reset()
        self.value = arr_in[0]

        if not self.fast:
            self._values.append(self.value)

        for value in arr_in[1:]:
            self.update(value)

    def update(self, new_val: float) -> None:
        """
        Updates the EMA calculator with a new data point.

        Parameters:
        -----------
        new_val : float
            The new data point to include in the EMA calculation.
        """
        self.value = self.alpha * new_val + (1.0 - self.alpha) * self.value
        if not self.fast:
            self._values.append(self.value)

    @property
    def values(self) -> np.ndarray:
        return self._values.as_array()

    def __eq__(self, ema: "ExponentialMovingAverage") -> bool:
        assert isinstance(ema, ExponentialMovingAverage)
        return np.all(ema.values == self.values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

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
    ringbuffer: RingBufferSingleDimFloat.class_type.instance_type

    def __init__(
        self, window: int, alpha: Optional[float] = None, fast: Optional[bool] = None
    ):
        self.window = window
        self.alpha = alpha if alpha is not None else 3.0 / (self.window + 1)
        self.fast = fast if fast is not None else True
        self.value = 0.0
        self.ringbuffer = RingBufferSingleDimFloat(self.window)

    def _recursive_ema(self, update: float) -> float:
        """
        Internal method to calculate the EMA given a new data point.

        Parameters:
        -----------
        update : float
            The new data point to include in the EMA calculation.

        Returns:
        --------
        float
            The updated EMA value.
        """
        return self.alpha * update + (1.0 - self.alpha) * self.value

    def as_array(self) -> np.ndarray[float]:
        """
        Compatibility with underlying ringbuffer for unwrapping.
        """
        return self.ringbuffer.as_array()

    def initialize(self, arr_in: np.ndarray[float]) -> None:
        """
        Initializes the EMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : Iterable[float]
            The initial series of data points to feed into the EMA calculator.
        """
        assert arr_in.ndim == 1
        self.ringbuffer.reset()
        self.value = arr_in[0]

        if not self.fast:
            self.ringbuffer.append(self.value)

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
        self.value = self._recursive_ema(new_val)
        if not self.fast:
            self.ringbuffer.append(self.value)

    def __eq__(self, ema: "ExponentialMovingAverage") -> bool:
        assert isinstance(ema, ExponentialMovingAverage)
        return ema.as_array() == self.as_array()

    def __len__(self) -> int:
        return len(self.ringbuffer)

    def __getitem__(self, index: int) -> float:
        return self.ringbuffer[index]

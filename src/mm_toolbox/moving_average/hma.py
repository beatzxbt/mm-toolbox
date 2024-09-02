import numpy as np
from numba.experimental import jitclass
from numba.types import bool_, uint32, float64

from mm_toolbox.moving_average.ema import ExponentialMovingAverage as EMA
from mm_toolbox.ringbuffer import RingBufferSingleDimFloat


@jitclass
class HullMovingAverage:
    window: uint32
    fast: bool_
    ringbuffer: RingBufferSingleDimFloat.class_type.instance_type
    value: float64
    
    _short_ema: EMA.class_type.instance_type
    _long_ema: EMA.class_type.instance_type
    _smooth_ema: EMA.class_type.instance_type

    def __init__(self, window: int, fast: bool = True):
        self.window = window
        self.fast = fast
        self.ringbuffer = RingBufferSingleDimFloat(window)
        self.value = 0.0

        self._short_ema = EMA(self.window // 2, 0.0, True)
        self._long_ema = EMA(window, 0.0, True)
        self._smooth_ema = EMA(int(window**0.5), 0.0, True)

    def _recursive_hma(self, value: float) -> float:
        """
        Internal method to calculate the HMA given a new data point.

        Parameters:
        -----------
        update : float
            The new data point to include in the HMA calculation.

        Returns:
        --------
        float
            The updated HMA value.
        """
        self._short_ema.update(value)
        self._long_ema.update(value)
        self._smooth_ema.update((self._short_ema.value * 2.0) - self._long_ema.value)
        return self._smooth_ema.value

    def as_array(self) -> np.ndarray[float]:
        """
        Compatibility with underlying ringbuffer for unwrapping.
        """
        return self.ringbuffer.as_array()

    def initialize(self, arr_in: np.ndarray[float]) -> None:
        """
        Initializes the HMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : np.ndarray[float]
            The initial series of data points to feed into the HMA calculator.
        """
        assert arr_in.ndim == 1
        self._short_ema.ringbuffer.reset()
        self._smooth_ema.ringbuffer.reset()
        self._long_ema.ringbuffer.reset()
        self.ringbuffer.reset()

        self.value = arr_in[0]
        for val in arr_in:
            self.update(val)

    def update(self, value: float) -> None:
        """
        Updates the HMA calculator with a new data point.

        Parameters:
        -----------
        new_val : float
            The new data point to include in the HMA calculation.
        """
        self.value = self._recursive_hma(value)
        if not self.fast:
            self.ringbuffer.append(self.value)

    def __eq__(self, ema: "HullMovingAverage") -> bool:
        assert isinstance(ema, HullMovingAverage)
        return ema.as_array() == self.as_array()

    def __len__(self) -> int:
        return len(self.ringbuffer)

    def __getitem__(self, index: int) -> float:
        return self.ringbuffer[index]

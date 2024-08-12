import numpy as np
from numba.experimental import jitclass
from numba.types import bool_, uint32, float64

from mm_toolbox.moving_average.ema import EMA
from mm_toolbox.ringbuffer import RingBufferSingleDimFloat

@jitclass
class HMA:
    window: uint32
    slow: bool_
    value: float64
    short_ema: EMA.class_type.instance_type
    long_ema: EMA.class_type.instance_type
    smooth_ema: EMA.class_type.instance_type
    rb: RingBufferSingleDimFloat.class_type.instance_type

    def __init__(self, window: int, fast: bool=True):
        self.window = window
        self.fast = fast
        self.short_ema = EMA(self.window // 2, 0, True)
        self.long_ema = EMA(window, 0, True)
        self.smooth_ema = EMA(int(window ** 0.5), 0, True)
        self.value = 0.0
        self.rb = RingBufferSingleDimFloat(window)

    def _reset_(self) -> None:
        """Clears the EMA & RB buffers."""
        self.rb.reset()
        self.short_ema.rb.reset()
        self.smooth_ema.rb.reset()
        self.long_ema.rb.reset()

    def _recursive_hma_(self, value: float) -> float:
        self.short_ema.update(value)
        self.long_ema.update(value)
        self.smooth_ema.update(self.short_ema.value * 2.0 - self.long_ema.value)
        return self.smooth_ema.value

    def initialize(self, arr_in: np.ndarray) -> None:
        self._reset_()
        self.value = arr_in[0]
        for val in arr_in:
            self.update(val)

    def update(self, value: float) -> None:
        self.value = self._recursive_hma_(value)
        if not self.fast:
            self.rb.appendright(self.value)
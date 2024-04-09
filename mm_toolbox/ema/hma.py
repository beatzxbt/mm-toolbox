import numpy as np
from numba.experimental import jitclass
from numba.types import bool_, int64, float64

from mm_toolbox.ema.ema import EMA
from mm_toolbox.ringbuffer.ringbuffer import RingBufferF64

spec_HMA_F64 = [
    ("window", int64),
    ("slow", bool_),
    ("value", float64),
    ("short_ema", EMA.class_type.instance_type),
    ("long_ema", EMA.class_type.instance_type),
    ("smooth_ema", EMA.class_type.instance_type),
    ("rb", RingBufferF64.class_type.instance_type),
]


@jitclass(spec_HMA_F64)
class HMA_F64:
    def __init__(self, window: int, fast: bool = True):
        self.window = window
        self.slow = not fast
        self.value = 0.0
        self.short_ema = EMA(int(self.window // 2), 0, True)
        self.long_ema = EMA(window, 0, True)
        self.smooth_ema = EMA(np.sqrt(window), 0, True)
        self.rb = RingBufferF64(window)

    def _recursive_hma_(self, value: float) -> float:
        self.short_ema.update(value)
        self.long_ema.update(value)
        self.smooth_ema.update(self.short_ema.value * 2 - self.long_ema.value)
        return self.smooth_ema.value

    def initialize(self, arr_in):
        self.rb.reset()
        self.short_ema.rb.reset()
        self.smooth_ema.rb.reset()
        self.long_ema.rb.reset()

        self.value = arr_in[0]

        for val in arr_in:
            self.update(val)

    def update(self, value: float):
        self.value = self._recursive_hma_(value)
        if self.slow:
            self.rb.appendright(self.value)


if __name__ == "__main__":
    # TEST
    arr = np.random.rand(10) * 1000

    hma = HMA_F64(window=20, fast=False)
    hma.initialize(arr)

    print(hma.value)
    print(hma.rb._unwrap_())

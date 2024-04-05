import numpy as np
from numba import njit
from numba.types import bool_, int64, float64, Array
from numba.experimental import jitclass
from typing import Optional

from mm_toolbox.ringbuffer.ringbuffer import RingBufferF64

spec = [
    ('window', int64),
    ('alpha', float64),
    ('fast', bool_),
    ('value', float64),
    ('rb', RingBufferF64.class_type.instance_type),
]

@jitclass(spec)
class EMA:
    def __init__(self, window: int, alpha: Optional[float]=0, fast: bool=False):
        self.window = window
        self.alpha = alpha if alpha != 0 else 2 / (self.window + 1)
        self.fast = fast
        self.value = 0.0
        self.rb = RingBufferF64(self.window) 

    def _recursive_ema_(self, update: float) -> float:
        return self.alpha * update + (1 - self.alpha) * self.value

    def initialize(self, arr_in):
        _ = self.rb.reset()
        self.value = arr_in[0]
        for val in arr_in:
            self.value = self._recursive_ema_(val)
            if not self.fast:
                self.rb.appendright(self.value)

    def update(self, new_val: float):
        self.value = self._recursive_ema_(new_val)
        if not self.fast:
            self.rb.appendright(self.value)


@njit(cache=True)
def ema_weights(window: int, reverse: bool=False, alpha: Optional[float]=0) -> Array:
    """
    Calculate EMA (Exponential Moving Average)-like weights for a given window size.

    Parameters
    ----------
    window : int
        The number of periods to use for the EMA calculation.
    reverse : bool, optional
        If True, the weights are returned in reverse order. The default is False.
    alpha : float, optional
        The decay factor for the EMA calculation. If not provided, it is calculated as 3 / (window + 1).

    Returns
    -------
    Array
        An array of EMA-like weights.

    Examples
    --------
    >>> ema_weights(window=5)
    array([0.33333333, 0.22222222, 0.14814815, 0.09876543, 0.06584362])

    >>> ema_weights(window=5, reverse=True)
    array([0.06584362, 0.09876543, 0.14814815, 0.22222222, 0.33333333])

    >>> ema_weights(window=5, alpha=0.5)
    array([0.5    , 0.25   , 0.125  , 0.0625 , 0.03125])
    """
    alpha = 3 / float(window + 1) if alpha == 0 else alpha
    weights = np.empty(window, dtype=float64)

    for i in range(window):
        weights[i] = alpha * (1 - alpha) ** i
 
    return weights[::-1] if reverse else weights
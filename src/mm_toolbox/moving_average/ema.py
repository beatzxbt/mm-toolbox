import numpy as np
from numba import njit
from numba.types import bool_, uint32, float64
from numba.experimental import jitclass
from typing import Optional

from mm_toolbox.ringbuffer import RingBufferSingleDimFloat

@jitclass
class EMA:
    """
    Exponential Moving Average (EMA) with optional RingBuffer to store history.

    Attributes:
    -----------
    window : int
        The window size for the EMA calculation.

    alpha : float
        The smoothing factor applied to the EMA. Default is calculated as `3 / (window + 1)`.

    fast : bool
        If True, the history of calculated EMA values is not stored.

    value : float
        The current value of the EMA.

    rb : RingBufferF64
        A ring buffer to store EMA values history, activated if `fast` is False.
    """
    
    window: uint32
    alpha: float64
    fast: bool_
    value: float64
    rb: RingBufferSingleDimFloat.class_type.instance_type

    def __init__(self, window: int, alpha: Optional[float]=0.0, fast: bool=True):
        self.window = window
        self.alpha = alpha if alpha != 0 else 3.0 / (self.window + 1)
        self.fast = fast
        self.value = 0.0
        self.rb = RingBufferSingleDimFloat(self.window)

    def _recursive_ema_(self, update: float) -> float:
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

    def initialize(self, arr_in: np.ndarray) -> None:
        """
        Initializes the EMA calculator with a series of data points.

        Parameters:
        -----------
        arr_in : Iterable[float]
            The initial series of data points to feed into the EMA calculator.
        """
        self.rb.reset()
        self.value = arr_in[0]
        
        if not self.fast:
            self.rb.appendright(self.value)

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
        self.value = self._recursive_ema_(new_val)
        if not self.fast:
            self.rb.appendright(self.value)



@njit(["float64[:](int64, float64)"], error_model="numpy", fastmath=True)
def ema_weights(window: int, alpha: Optional[float]=0.0) -> np.ndarray:
    """
    Calculate EMA (Exponential Moving Average)-like weights for a given window size.

    Parameters
    ----------
    window : int
        The number of periods to use for the EMA calculation.

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

    >>> ema_weights(window=5, alpha=0.5)
    array([0.5    , 0.25   , 0.125  , 0.0625 , 0.03125])
    """
    alpha = 3.0 / float(window + 1) if alpha == 0.0 else alpha
    weights = np.empty(window, dtype=float64)

    for i in range(window):
        weights[i] = alpha * (1.0 - alpha) ** i
 
    return weights
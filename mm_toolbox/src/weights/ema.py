import numpy as np
from numba import njit
from numba.types import float64
from typing import Optional

@njit(["float64[:](int64, float64)"], fastmath=True)
def ema_weights(window: int, alpha: Optional[float]=0.0) -> np.ndarray[float]:
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
    np.ndarray
        An array of EMA-like weights.

    Examples
    --------
    >>> ema_weights(window=5)
    array([0.33333333, 0.22222222, 0.14814815, 0.09876543, 0.06584362])

    >>> ema_weights(window=5, alpha=0.5)
    array([0.5    , 0.25   , 0.125  , 0.0625 , 0.03125])
    """
    assert window > 1
    alpha = 3.0 / float(window + 1) if alpha == 0.0 else alpha
    weights = np.array([alpha * (1.0 - alpha) ** i for i in range(window)], dtype=float64)
    return weights
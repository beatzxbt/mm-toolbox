import numpy as np
from numba import njit
from numba.types import float64
from typing import Optional

@njit(["float64[:](int64, float64)"], error_model="numpy")
def geometric_weights(num: int, r: Optional[float] = 0.75) -> np.ndarray:
    """
    Generates a list of `num` weights that follow a geometric distribution and sum to 1.

    Parameters
    ----------
    num : int
        The number of weights to generate.

    r : float, optional
        The common ratio of the geometric sequence. Must be strictly between 0 and 1. The default value is 0.75.

    Returns
    -------
    np.ndarray
        An array of weights, following a geometric distribution, whose sum is 1.
    """
    assert num > 1
    weights = np.array([r**i for i in range(num)], dtype=float64)
    normalized = weights / weights.sum()
    return normalized
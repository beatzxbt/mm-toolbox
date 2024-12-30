import numpy as np
from typing import Optional


def geometric_weights(num: int, r: Optional[float] = None) -> np.ndarray:
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
        An array of normalized geometric weights from lowest -> highest.
    """
    if num <= 1:
        raise ValueError("Number of weights generated cannot be <1.")

    r = r if r else 0.75
    weights = np.array([r**i for i in range(num)], dtype=np.float64)
    normalized = weights / weights.sum()
    return normalized

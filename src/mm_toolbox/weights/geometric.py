import numpy as np
from typing import Optional


def geometric_weights(
    num: int, r: Optional[float] = None, normalized: bool = True
) -> np.ndarray:
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
        raise ValueError(f"Invalid number of weights; expected > 1 but got {num}.")

    r = r if r else 0.75
    weights = np.array([r**i for i in range(num)], dtype=np.float64)
    if normalized:
        return weights / weights.sum()
    return weights

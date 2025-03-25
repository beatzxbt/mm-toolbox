import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def logarithmic_weights(num: int) -> np.ndarray:
    """
    Generates a list of `num` weights that follow a logarithmic distribution and sum to 1.

    Parameters
    ----------
    num : int
        The number of weights to generate.

    Returns
    -------
    np.ndarray
        An array of normalized logarithmic weights from lowest -> highest.
    """
    if num <= 1:
        raise ValueError(f"Invalid number of weights; expected > 1 but got {num}.")

    # Start from 1 to avoid log(0)
    weights = np.log(np.arange(1, num + 1, dtype=np.float64))
    normalized = weights / weights.sum()
    return normalized

"""Logarithmic weight calculations."""

import numpy as np
from numpy.typing import NDArray


def logarithmic_weights(num: int, normalized: bool = True) -> NDArray[np.float64]:
    """Return logarithmic weights of length ``num`` (start from log(1))."""
    if num <= 1:
        raise ValueError(f"Invalid number of weights; expected > 1 but got {num}.")

    # Start from 1 to avoid log(0)
    weights = np.log(np.arange(1, num + 1, dtype=np.float64))
    if normalized:
        return weights / weights.sum()
    return weights

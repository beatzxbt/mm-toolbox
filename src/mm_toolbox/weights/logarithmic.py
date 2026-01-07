"""Logarithmic weight calculations."""

import numpy as np
from numpy.typing import NDArray


def logarithmic_weights(num: int, normalized: bool = True) -> NDArray[np.float64]:
    """Return logarithmic weights of length ``num``.

    Args:
        num: Number of weights to generate. Must be > 1.
        normalized: Whether to normalize weights so they sum to 1.

    Returns:
        Array of logarithmic weights starting at log(1).

    Raises:
        ValueError: If ``num`` is <= 1.
    """
    if num <= 1:
        raise ValueError(f"Invalid number of weights; expected > 1 but got {num}.")

    # Start from 1 to avoid log(0)
    weights = np.log(np.arange(1, num + 1, dtype=np.float64))
    if normalized:
        return weights / weights.sum()
    return weights

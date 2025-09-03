"""Geometric weight calculations."""

import numpy as np
from numpy.typing import NDArray


def geometric_weights(
    num: int, r: float | None = None, normalized: bool = True
) -> NDArray[np.float64]:
    """Return geometric weights of length ``num`` with ratio ``r`` (default 0.75)."""
    if num <= 1:
        raise ValueError(f"Invalid number of weights; expected > 1 but got {num}.")

    r = r if r is not None else 0.75
    weights = np.array([r**i for i in range(num)], dtype=np.float64)
    if normalized:
        return weights / weights.sum()
    return weights

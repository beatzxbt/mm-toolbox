"""Geometric weight calculations."""

import numpy as np
from numpy.typing import NDArray


def geometric_weights(
    num: int, r: float | None = None, normalized: bool = True
) -> NDArray[np.float64]:
    """Return geometric weights of length ``num``.

    Args:
        num: Number of weights to generate. Must be > 1.
        r: Common ratio for the geometric series. Defaults to 0.75 when ``None``.
        normalized: Whether to normalize weights so they sum to 1.

    Returns:
        Array of geometric weights in ascending exponent order.

    Raises:
        ValueError: If ``num`` is <= 1.
    """
    if num <= 1:
        raise ValueError(f"Invalid number of weights; expected > 1 but got {num}.")

    r = r if r is not None else 0.75
    weights = np.power(np.float64(r), np.arange(num, dtype=np.float64))
    if normalized:
        return weights / weights.sum()
    return weights

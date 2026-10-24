"""Exponential moving average (EMA) weight calculations.

Provides ``ema_weights`` for generating normalized or raw EMA-like
weight arrays over a fixed window.
"""

from math import isfinite

import numpy as np
from numpy.typing import NDArray


def ema_weights(
    window: int, alpha: float | None = None, normalized: bool = True
) -> NDArray[np.float64]:
    """Return EMA-like weights of length ``window``.

    Args:
        window: Number of weights to generate. Must be > 1.
        alpha: Smoothing factor. Defaults to ``2/(window+1)`` when ``None``.
        normalized: Whether to normalize weights so they sum to 1.

    Returns:
        Array of EMA weights in chronological order (oldest to newest).

    Raises:
        ValueError: If ``window`` is <= 1 or ``alpha`` is outside [0, 1].
    """
    if window <= 1:
        raise ValueError(f"Invalid window size; expected > 1 but got {window}.")

    alpha = alpha if alpha is not None else 2.0 / float(window + 1)
    if not isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")

    if normalized and alpha == 0.0:
        # This is the finite limit of the normalized geometric weights as
        # alpha approaches zero: every position has equal weight.
        return np.full(window, 1.0 / window, dtype=np.float64)

    exponents = np.arange(window - 1, -1, -1, dtype=np.float64)
    weights = alpha * np.power(1.0 - alpha, exponents)
    if normalized:
        return weights / weights.sum()
    return weights

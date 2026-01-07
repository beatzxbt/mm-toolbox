"""Exponential moving average weight calculations."""

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
        ValueError: If ``window`` is <= 1.
    """
    if window <= 1:
        raise ValueError(f"Invalid window size; expected > 1 but got {window}.")

    alpha = alpha if alpha is not None else 2.0 / float(window + 1)

    weights = np.array(
        [alpha * (1.0 - alpha) ** i for i in range(window - 1, -1, -1)],
        dtype=np.float64,
    )
    if normalized:
        return weights / weights.sum()
    return weights

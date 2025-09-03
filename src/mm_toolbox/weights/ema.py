"""Exponential moving average weight calculations."""

import numpy as np
from numpy.typing import NDArray


def ema_weights(
    window: int, alpha: float | None = None, normalized: bool = True
) -> NDArray[np.float64]:
    """Return EMA-like weights of length ``window`` using ``alpha`` or default 3/(window+1)."""
    if window <= 1:
        raise ValueError(f"Invalid window size; expected > 1 but got {window}.")

    alpha = alpha if alpha is not None else 3.0 / float(window + 1)

    weights = np.array(
        [alpha * (1.0 - alpha) ** i for i in range(window - 1, -1, -1)],
        dtype=np.float64,
    )
    if normalized:
        return weights / weights.sum()
    return weights

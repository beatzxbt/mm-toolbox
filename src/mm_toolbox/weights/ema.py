import numpy as np
from typing import Optional

def ema_weights(window: int, alpha: Optional[float] = None, normalized: bool=True) -> np.ndarray:
    """
    Calculate EMA (Exponential Moving Average)-like weights for a given window size.

    Parameters
    ----------
    window : int
        The number of periods to use for the EMA calculation.

    alpha : float, optional
        The decay factor for the EMA calculation. If not provided, it is calculated as 3 / (window + 1).

    Returns
    -------
    np.ndarray
        An array of normalized EMA-like weights from lowest -> highest.
    """
    if window <= 1:
        raise ValueError(f"Invalid window size; expected > 1 but got {window}.")

    alpha = alpha if alpha else 3.0 / float(window + 1)

    weights = np.array(
        [alpha * (1.0 - alpha) ** i for i in range(window - 1, -1, -1)],
        dtype=np.float64,
    )
    if normalized:
        return weights / weights.sum()
    return weights

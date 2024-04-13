from numba import njit, float64

@njit(["float64(float64, float64, float64)"], cache=True)
def true_range(open: float, high: float, low: float) -> float:
    """
    Calculate the true range of a trading price bar.

    The true range is the greatest of the following:
    - The difference between the current high and the current low,
    - The absolute difference between the current high and the previous close,
    - The absolute difference between the current low and the previous close.

    Parameters
    ----------
    open : float
        The opening price of the current period.
        
    high : float
        The highest price of the current period.
        
    low : float
        The lowest price of the current period.

    Returns
    -------
    float
        The true range of the price for the period.

    Examples
    --------
    >>> true_range(100.0, 105.0, 95.0)
    10.0

    Note that the function assumes 'open' as the previous close for calculation purposes.
    """
    return max(max(open - low, abs(high - open)), abs(low - open))

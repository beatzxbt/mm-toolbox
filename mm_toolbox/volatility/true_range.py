import numpy as np
from numba import njit, int64, float64


@njit(float64(float64, float64, float64), nogil=True)
def calculate_true_range_f64(open_value: float, high_value: float, low_value: float) -> float:
    return np.maximum(
        np.maximum(open_value - low_value, np.abs(high_value - open_value)), np.abs(low_value - open_value)
    )


@njit(int64(int64, int64, int64), nogil=True)
def calculate_true_range_i64(open_value: int, high_value: int, low_value: int) -> int:
    return np.maximum(
        np.maximum(open_value - low_value, np.abs(high_value - open_value)), np.abs(low_value - open_value)
    )


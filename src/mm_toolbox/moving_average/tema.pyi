from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .base import MovingAverage

__all__ = ["TimeExponentialMovingAverage"]

class TimeExponentialMovingAverage(MovingAverage):
    def __init__(
        self, window: int = 2, is_fast: bool = False, half_life_s: float = 1.0
    ) -> None: ...
    def initialize(self, values: npt.NDArray[np.float64]) -> float: ...
    def next(self, new_val: float) -> float: ...
    def update(self, new_val: float) -> float: ...

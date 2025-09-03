from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .base import MovingAverage

__all__ = ["ExponentialMovingAverage"]

class ExponentialMovingAverage(MovingAverage):
    def __init__(
        self, window: int, is_fast: bool = False, alpha: float = 0.0
    ) -> None: ...
    def initialize(self, values: npt.NDArray[np.float64]) -> float: ...
    def next(self, new_val: float) -> float: ...
    def update(self, new_val: float) -> float: ...

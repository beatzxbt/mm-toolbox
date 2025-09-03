from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .base import MovingAverage

__all__ = ["WeightedMovingAverage"]

class WeightedMovingAverage(MovingAverage):
    def __init__(self, window: int, fast: bool = False) -> None: ...
    def initialize(self, values: npt.NDArray[np.float64]) -> float: ...
    def next(self, new_val: float) -> float: ...
    def update(self, new_val: float) -> float: ...

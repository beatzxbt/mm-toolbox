from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MovingAverageProtocol(Protocol):
    """Protocol unifying all moving average implementations."""

    def initialize(self, values: npt.NDArray[np.float64]) -> float:
        """Initialize the moving average with the given values."""
        ...

    def next(self, new_val: float) -> float:
        """Calculate the next value without updating internal state."""
        ...

    def update(self, new_val: float) -> float:
        """Update the moving average with the next value."""
        ...

    def get_value(self) -> float:
        """Return the current moving average value."""
        ...

    def get_values(self) -> npt.NDArray[np.float64]:
        """Return historical values as a NumPy array."""
        ...

    def __len__(self) -> int:
        """Return the number of stored values."""
        ...

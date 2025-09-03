from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["Rounder", "RounderConfig"]

class RounderConfig:
    tick_size: float
    lot_size: float
    round_bids_down: bool
    round_asks_up: bool
    round_size_up: bool

    def __init__(
        self,
        *,
        tick_size: float,
        lot_size: float,
        round_bids_down: bool,
        round_asks_up: bool,
        round_size_up: bool,
    ) -> None: ...
    @classmethod
    def default(cls, tick_size: float, lot_size: float) -> RounderConfig: ...

class Rounder:
    """Rounding operations for prices and sizes.

    Args:
        tick_size: Minimum price increment.
        lot_size: Minimum size increment.

    """

    tick_size: float
    lot_size: float

    def __init__(self, config: RounderConfig) -> None: ...
    def bid(self, price: float) -> float:
        """Round a price down to the nearest tick size multiple."""
        ...

    def ask(self, price: float) -> float:
        """Round a price up to the nearest tick size multiple."""
        ...

    def size(self, size: float) -> float:
        """Round a size down to the nearest lot size multiple."""
        ...

    def bids(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Round an array of prices down to the nearest tick size multiple."""
        ...

    def asks(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Round an array of prices up to the nearest tick size multiple."""
        ...

    def sizes(self, sizes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Round an array of sizes down to the nearest lot size multiple."""
        ...

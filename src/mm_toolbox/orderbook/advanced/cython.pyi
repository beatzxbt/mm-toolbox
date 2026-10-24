"""Cython-facing advanced orderbook wrapper."""

from __future__ import annotations

from .enum.enums import CyOrderbookSortedness

class CyAdvancedOrderbook:
    """Cython-facing wrapper around the internal advanced orderbook core."""

    def __init__(
        self,
        tick_size: float,
        lot_size: float,
        num_levels: int,
        delta_sortedness: CyOrderbookSortedness = ...,
        snapshot_sortedness: CyOrderbookSortedness = ...,
    ) -> None:
        """Initialize the Cython-facing advanced orderbook.

        Args:
            tick_size: Minimum price increment. Must be positive.
            lot_size: Minimum size increment. Must be positive.
            num_levels: Maximum number of levels per side. Must be at least 4.
            delta_sortedness: Expected delta order, or UNKNOWN for lazy inference.
            snapshot_sortedness: Expected snapshot order, or UNKNOWN for lazy inference.
        """
        ...

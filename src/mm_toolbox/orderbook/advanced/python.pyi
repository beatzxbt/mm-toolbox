"""Type stubs for python.pyx - Python-facing advanced orderbook API."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from .enum.enums import PyOrderbookSortedness
from .level.level import PyOrderbookLevel, PyOrderbookLevels

class PyAdvancedOrderbook:
    """Python-facing API for the advanced orderbook with efficient in-place updates.

    This class provides a Python API wrapper around CoreAdvancedOrderbook, designed
    for use in Python code. It handles conversion between Python types (PyOrderbookLevel,
    PyOrderbookLevels) and the internal C structs.

    The orderbook maintains separate bid and ask sides, each with a fixed maximum
    number of levels. Levels are stored internally using integer arithmetic (ticks
    and lots) for precision and performance.
    """

    def __init__(
        self,
        tick_size: float,
        lot_size: float,
        num_levels: int,
        delta_sortedness: Optional[PyOrderbookSortedness] = None,
        snapshot_sortedness: Optional[PyOrderbookSortedness] = None,
    ) -> None:
        """Initialize a new PyAdvancedOrderbook instance.

        Args:
            tick_size: Minimum price increment (must be > 0)
            lot_size: Minimum size increment (must be > 0)
            num_levels: Maximum number of levels per side (must be > 0)
            delta_sortedness: Expected sort order for delta updates (default: UNKNOWN)
            snapshot_sortedness: Expected sort order for snapshot updates (default: UNKNOWN)
        """
        ...

    def clear(self) -> None:
        """Clear all levels from both sides of the orderbook."""
        ...

    def consume_snapshot(
        self, asks: PyOrderbookLevels, bids: PyOrderbookLevels
    ) -> None:
        """Replace the entire orderbook state with new snapshot data.

        Args:
            asks: PyOrderbookLevels containing ask levels
            bids: PyOrderbookLevels containing bid levels
        """
        ...

    def consume_deltas(self, asks: PyOrderbookLevels, bids: PyOrderbookLevels) -> None:
        """Apply incremental updates to the orderbook.

        Args:
            asks: PyOrderbookLevels containing ask level updates
            bids: PyOrderbookLevels containing bid level updates
        """
        ...

    def consume_bbo(self, ask: PyOrderbookLevel, bid: PyOrderbookLevel) -> None:
        """Update only the best bid and offer (top of book).

        Args:
            ask: PyOrderbookLevel for the best ask
            bid: PyOrderbookLevel for the best bid
        """
        ...

    def consume_snapshot_numpy(
        self,
        ask_prices: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
        bid_prices: npt.NDArray[np.float64],
        bid_sizes: npt.NDArray[np.float64],
        ask_norders: Optional[npt.NDArray[np.uint64]] = None,
        bid_norders: Optional[npt.NDArray[np.uint64]] = None,
    ) -> None:
        """Replace the entire orderbook state with new snapshot data from numpy arrays.

        This method provides a more efficient path for users who already have data
        in numpy arrays, avoiding the overhead of constructing PyOrderbookLevels.

        Args:
            ask_prices: 1D array of ask prices
            ask_sizes: 1D array of ask sizes (must match length of ask_prices)
            bid_prices: 1D array of bid prices
            bid_sizes: 1D array of bid sizes (must match length of bid_prices)
            ask_norders: Optional 1D array of ask norders (defaults to 1 for each level)
            bid_norders: Optional 1D array of bid norders (defaults to 1 for each level)
        """
        ...

    def consume_deltas_numpy(
        self,
        ask_prices: npt.NDArray[np.float64],
        ask_sizes: npt.NDArray[np.float64],
        bid_prices: npt.NDArray[np.float64],
        bid_sizes: npt.NDArray[np.float64],
        ask_norders: Optional[npt.NDArray[np.uint64]] = None,
        bid_norders: Optional[npt.NDArray[np.uint64]] = None,
    ) -> None:
        """Apply incremental updates to the orderbook from numpy arrays.

        This method provides a more efficient path for users who already have data
        in numpy arrays, avoiding the overhead of constructing PyOrderbookLevels.

        Args:
            ask_prices: 1D array of ask prices
            ask_sizes: 1D array of ask sizes (must match length of ask_prices)
            bid_prices: 1D array of bid prices
            bid_sizes: 1D array of bid sizes (must match length of bid_prices)
            ask_norders: Optional 1D array of ask norders (defaults to 1 for each level)
            bid_norders: Optional 1D array of bid norders (defaults to 1 for each level)
        """
        ...

    def get_bbo(self) -> tuple[PyOrderbookLevel, PyOrderbookLevel]:
        """Get the best bid and offer (top of book).

        Returns:
            Tuple of (best_bid, best_ask) as PyOrderbookLevel instances

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def get_bids(self) -> PyOrderbookLevels:
        """Get all bid levels."""
        ...

    def get_asks(self) -> PyOrderbookLevels:
        """Get all ask levels."""
        ...

    def get_bids_numpy(self) -> npt.NDArray[np.void]:
        """Get bid levels as a NumPy structured array."""
        ...

    def get_asks_numpy(self) -> npt.NDArray[np.void]:
        """Get ask levels as a NumPy structured array."""
        ...

    def get_mid_price(self) -> float:
        """Calculate the mid price (average of best bid and ask).

        Returns:
            Mid price

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def get_bbo_spread(self) -> float:
        """Calculate the spread between best bid and ask.

        Returns:
            Spread in price units

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def get_wmid_price(self) -> float:
        """Calculate the weighted mid price (volume-weighted average of best bid and ask).

        Returns:
            Weighted mid price

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def get_volume_weighted_mid_price(
        self, size: float, is_base_currency: bool = True
    ) -> float:
        """Calculate volume-weighted mid price for a given trade size.

        Args:
            size: Trade size to calculate weighted price for
            is_base_currency: If True, size is in base currency; if False, in quote currency

        Returns:
            Volume-weighted mid price, or infinity if size cannot be filled

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def get_price_impact(
        self, size: float, is_buy: bool, is_base_currency: bool = True
    ) -> float:
        """Calculate price impact of executing a trade of given size.

        Args:
            size: Trade size
            is_buy: If True, anchor at best ask and consume asks upward; if False, anchor at best bid and consume bids downward
            is_base_currency: If True, size is in base currency; if False, convert quote size to base using the same touch anchor price

        Returns:
            Absolute terminal impact from touch anchor to last consumed level, or infinity if size cannot be filled

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def get_size_for_price_impact_bps(
        self, impact_bps: float, is_buy: bool, is_base_currency: bool = True
    ) -> float:
        """Calculate cumulative size available within a basis-point depth band.

        Args:
            impact_bps: Price depth in basis points from touch
            is_buy: If True, aggregate asks up to best_ask * (1 + impact_bps/10000); if False, bids down to best_bid * (1 - impact_bps/10000)
            is_base_currency: If True, return base size; if False, return quote notional

        Returns:
            Cumulative available size within the band

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def is_bbo_crossed(self, bid_price: float, ask_price: float) -> bool:
        """Check if this orderbook's BBO crosses with another orderbook's BBO.

        Args:
            bid_price: Best bid price from another orderbook
            ask_price: Best ask price from another orderbook

        Returns:
            True if the BBOs cross (this bid >= other ask or this ask <= other bid)

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

    def does_bbo_price_change(self, bid_price: float, ask_price: float) -> bool:
        """Check if the given prices differ from current BBO.

        Args:
            bid_price: Bid price to compare
            ask_price: Ask price to compare

        Returns:
            True if either price differs from current BBO

        Raises:
            RuntimeError: If orderbook is empty
        """
        ...

"""Shared fixtures and helpers for orderbook tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_toolbox.orderbook.advanced import (
    AdvancedOrderbook,
    OrderbookLevel,
    OrderbookLevels,
    PyOrderbookSortedness,
)

# Aliases for backward compatibility in test code
PyAdvancedOrderbook = AdvancedOrderbook
PyOrderbookLevel = OrderbookLevel
PyOrderbookLevels = OrderbookLevels

TICK_SIZE = 0.01
LOT_SIZE = 0.001


def _make_levels(
    prices: list[float],
    sizes: list[float],
    norders: list[int] | None = None,
    *,
    with_precision: bool,
) -> tuple[PyOrderbookLevels, bool]:
    """Create OrderbookLevels from price/size lists."""
    norders = [1] * len(prices) if norders is None else norders
    if with_precision:
        return (
            PyOrderbookLevels.from_list_with_ticks_and_lots(
                prices, sizes, norders, TICK_SIZE, LOT_SIZE
            ),
            True,
        )
    return PyOrderbookLevels.from_list(prices, sizes, norders), False


def _empty_levels() -> PyOrderbookLevels:
    """A no-op delta payload for asks that won't affect a typical test orderbook.

    Uses price=999.0 which is far ABOVE the test orderbook asks (~100.0).
    With size=0 and being outside the ask range, this level will be ignored.
    """
    return PyOrderbookLevels.from_list_with_ticks_and_lots(
        [999.0], [0.0], [0], TICK_SIZE, LOT_SIZE
    )


def _empty_bid_levels() -> PyOrderbookLevels:
    """A no-op delta payload for bids that won't affect a typical test orderbook.

    Uses price=1.0 which is far BELOW the test orderbook bids (~100.0).
    With size=0 and being outside the bid range, this level will be ignored.
    """
    return PyOrderbookLevels.from_list_with_ticks_and_lots(
        [1.0], [0.0], [0], TICK_SIZE, LOT_SIZE
    )


def _mk_book(num_levels: int = 64) -> PyAdvancedOrderbook:
    """Create a test orderbook with default settings."""
    return PyAdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=num_levels,
        delta_sortedness=PyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
    )


def _bids_asks_arrays(book: PyAdvancedOrderbook) -> tuple[np.ndarray, np.ndarray]:
    """Get bids and asks as numpy arrays."""
    return np.asarray(book.get_bids_numpy()), np.asarray(book.get_asks_numpy())


def _bbo_prices(book: PyAdvancedOrderbook) -> tuple[float, float]:
    """Get best bid and ask prices."""
    bids, asks = _bids_asks_arrays(book)
    return float(bids["price"][0]), float(asks["price"][0])


@pytest.fixture
def max_capacity_orderbook() -> PyAdvancedOrderbook:
    """Orderbook at maximum practical capacity (1000 levels)."""
    return _mk_book(num_levels=1000)


@pytest.fixture
def pathological_data() -> dict[str, tuple[float, float] | float]:
    """Generator for pathological market data scenarios."""
    return {
        "huge_spread": (1.0, 10000.0),  # bid=1.0, ask=10000.0
        "zero_spread": (100.0, 100.0),  # bid=ask=100.0
        "negative_spread": (100.01, 100.0),  # crossed orderbook
        "extreme_price_high": 1e100,
        "extreme_price_low": 1e-100,
        "extreme_size": 1e15,
    }

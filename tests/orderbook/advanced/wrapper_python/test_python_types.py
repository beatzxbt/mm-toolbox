"""Tests for raw OrderbookLevel and OrderbookLevels wrappers.

Layer 1 primitive tests: validate PyOrderbookLevel creation, validation,
and OrderbookLevels factory methods.
"""

from __future__ import annotations

import pytest

from mm_toolbox.orderbook.advanced import OrderbookLevels
from tests.orderbook.advanced.conftest import (
    PyOrderbookLevel,
    _mk_book,
)


class TestPyOrderbookLevel:
    """Layer 1: Validate PyOrderbookLevel creation."""

    def test_basic_creation_and_validation(self):
        """Given valid price/size/norders, When creating level, Then fields are set correctly."""
        level = PyOrderbookLevel(price=100.0, size=1.5, norders=2, verify_values=True)
        assert level.price == pytest.approx(100.0)
        assert level.size == pytest.approx(1.5)
        assert level.norders == 2

    def test_invalid_values_raise(self):
        """Given invalid values, When creating level, Then appropriate errors raised."""
        with pytest.raises(ValueError):
            PyOrderbookLevel(price=-1.0, size=1.0, norders=1)
        with pytest.raises(ValueError):
            PyOrderbookLevel(price=1.0, size=-1.0, norders=1)
        # norders is u64, so negative values cause OverflowError at the Cython type conversion level
        with pytest.raises((ValueError, OverflowError)):
            PyOrderbookLevel(price=1.0, size=1.0, norders=-1)


class TestPyOrderbookLevels:
    """Layer 1: Validate PyOrderbookLevels factories and defaults."""

    def test_from_list_defaults_norders(self):
        """Given prices and sizes without norders, When consumed, Then norders defaults to 1."""
        asks = OrderbookLevels.from_list([100.01], [1.5])
        bids = OrderbookLevels.from_list([100.00], [1.0])
        book = _mk_book(num_levels=64)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert asks_arr["norders"][0] == 1
        assert bids_arr["norders"][0] == 1

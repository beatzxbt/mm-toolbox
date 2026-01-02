"""Tests for OrderbookLevel and utility conversion functions."""

from __future__ import annotations

import pytest

from mm_toolbox.orderbook.advanced import (
    OrderbookLevels,
    convert_price_from_tick,
    convert_price_to_tick,
    convert_size_from_lot,
    convert_size_to_lot,
)
from tests.orderbook.advanced.conftest import (
    LOT_SIZE,
    PyOrderbookLevel,
    TICK_SIZE,
    _mk_book,
)


class TestUtilityFunctions:
    """Test the utility conversion helpers."""

    def test_price_to_ticks_conversion(self):
        assert convert_price_to_tick(100.01, TICK_SIZE) == 10001
        assert convert_price_to_tick(100.0, TICK_SIZE) == 10000
        assert convert_price_to_tick(99.99, TICK_SIZE) == 9999
        assert convert_price_to_tick(100.125, 0.125) == 801
        assert convert_price_to_tick(100.0001, 0.0001) == 1000001

    def test_size_to_lots_conversion(self):
        assert convert_size_to_lot(1.0, LOT_SIZE) == 1000
        assert convert_size_to_lot(0.5, LOT_SIZE) == 500
        assert convert_size_to_lot(0.0001, 0.0001) == 1

    def test_price_from_ticks_conversion(self):
        assert convert_price_from_tick(10001, TICK_SIZE) == 100.01
        assert convert_price_from_tick(10000, TICK_SIZE) == 100.0
        assert abs(convert_price_from_tick(801, 0.125) - 100.125) < 1e-10

    def test_size_from_lots_conversion(self):
        assert convert_size_from_lot(1000, LOT_SIZE) == 1.0
        assert convert_size_from_lot(500, LOT_SIZE) == 0.5
        assert abs(convert_size_from_lot(1, 0.0001) - 0.0001) < 1e-10


class TestPyOrderbookLevel:
    """Validate PyOrderbookLevel creation and helpers."""

    def test_basic_creation_and_validation(self):
        level = PyOrderbookLevel(price=100.0, size=1.5, norders=2, verify_values=True)
        assert level.price == pytest.approx(100.0)
        assert level.size == pytest.approx(1.5)
        assert level.norders == 2
        assert level.ticks == 0
        assert level.lots == 0

    def test_invalid_values_raise(self):
        with pytest.raises(ValueError):
            PyOrderbookLevel(price=-1.0, size=1.0, norders=1)
        with pytest.raises(ValueError):
            PyOrderbookLevel(price=1.0, size=-1.0, norders=1)
        # norders is u64, so negative values cause OverflowError at the Cython type conversion level
        with pytest.raises((ValueError, OverflowError)):
            PyOrderbookLevel(price=1.0, size=1.0, norders=-1)

    def test_with_ticks_and_lots(self):
        level = PyOrderbookLevel.with_ticks_and_lots(
            price=100.01, size=1.5, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=2
        )
        assert level.ticks == convert_price_to_tick(100.01, TICK_SIZE)
        assert level.lots == convert_size_to_lot(1.5, LOT_SIZE)


class TestPyOrderbookLevels:
    """Validate PyOrderbookLevels factories and defaults."""

    def test_from_list_defaults_norders(self):
        asks = OrderbookLevels.from_list([100.01], [1.5])
        bids = OrderbookLevels.from_list([100.00], [1.0])
        book = _mk_book(num_levels=64)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert asks_arr["norders"][0] == 1
        assert bids_arr["norders"][0] == 1

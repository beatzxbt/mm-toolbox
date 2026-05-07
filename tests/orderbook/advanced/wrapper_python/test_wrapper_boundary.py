"""Tests for Python wrapper boundary behavior."""

from __future__ import annotations

import pytest
from mm_toolbox.orderbook.advanced import AdvancedOrderbook
from tests.orderbook.advanced.conftest import LOT_SIZE


class TestPythonErrorMessages:
    """Verify error messages are helpful."""

    def test_empty_book_error_message(self, empty_book):
        with pytest.raises(RuntimeError, match="Empty view"):
            empty_book.get_mid_price()

    def test_invalid_init_error_message(self):
        with pytest.raises(ValueError, match="Invalid tick_size"):
            AdvancedOrderbook(
                tick_size=-0.01,
                lot_size=LOT_SIZE,
                num_levels=64,
            )


class TestCpdefBehavior:
    """Verify cpdef methods work correctly from Python."""

    def test_method_is_callable_from_python(self, standard_book):
        mid = standard_book.get_mid_price()
        assert isinstance(mid, float)

    def test_numpy_arrays_accepted(self, empty_book):
        import numpy as np

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        empty_book.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)
        assert empty_book.get_mid_price() > 0

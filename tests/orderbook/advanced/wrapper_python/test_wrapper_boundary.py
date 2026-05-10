"""Tests for the Python wrapper boundary behavior of AdvancedOrderbook.

Validates that error messages are user-friendly from Python, cpdef methods
are callable, and numpy arrays are accepted as inputs.
"""

from __future__ import annotations

import pytest
from mm_toolbox.orderbook.advanced import AdvancedOrderbook
from tests.orderbook.advanced.conftest import LOT_SIZE


class TestPythonErrorMessages:
    """Layer 2: Verify Python-facing error messages are helpful."""

    def test_empty_book_error_message(self, empty_book):
        """Given an empty book, When get_mid_price is called, Then raises with 'Empty view'."""
        with pytest.raises(RuntimeError, match="Empty view"):
            empty_book.get_mid_price()

    def test_invalid_init_error_message(self):
        """Given invalid tick_size, When creating AdvancedOrderbook, Then raises with helpful message."""
        with pytest.raises(ValueError, match="Invalid tick_size"):
            AdvancedOrderbook(
                tick_size=-0.01,
                lot_size=LOT_SIZE,
                num_levels=64,
            )


class TestCpdefBehavior:
    """Layer 2: Verify cpdef methods work correctly from Python."""

    def test_method_is_callable_from_python(self, standard_book):
        """Given a populated book, When get_mid_price is called from Python, Then returns a float."""
        mid = standard_book.get_mid_price()
        assert isinstance(mid, float)

    def test_numpy_arrays_accepted(self, empty_book):
        """Given numpy arrays, When consume_snapshot_numpy is called, Then book is populated."""
        import numpy as np

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        empty_book.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)
        assert empty_book.get_mid_price() > 0

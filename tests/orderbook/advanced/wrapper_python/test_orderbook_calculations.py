"""Tests for price impact and size calculations on the advanced orderbook."""

from __future__ import annotations

import pytest

from tests.orderbook.advanced.conftest import (
    _mk_book,
    _make_levels,
)


class TestPriceImpactBps:
    """Tests for get_size_for_price_impact_bps."""

    def test_buy_base_currency_single_level(self):
        """Impact band covers two ask levels, returning total base size."""
        book = _mk_book()
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.0, 2.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.00],
            sizes=[1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        result = book.get_size_for_price_impact_bps(
            100.0, is_buy=True, is_base_currency=True
        )
        assert result == pytest.approx(3.0)

    def test_sell_base_currency(self):
        """Sell side impact band covers one bid level."""
        book = _mk_book()
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.0, 2.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.00],
            sizes=[1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        result = book.get_size_for_price_impact_bps(
            100.0, is_buy=False, is_base_currency=True
        )
        assert result == pytest.approx(1.0)

    def test_zero_impact_bps(self):
        """Zero impact returns 0.0."""
        book = _mk_book()
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.0, 2.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.00],
            sizes=[1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        result = book.get_size_for_price_impact_bps(
            0.0, is_buy=True, is_base_currency=True
        )
        assert result == 0.0

    def test_negative_impact_bps(self):
        """Negative impact returns 0.0."""
        book = _mk_book()
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.0, 2.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.00],
            sizes=[1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        result = book.get_size_for_price_impact_bps(
            -1.0, is_buy=True, is_base_currency=True
        )
        assert result == 0.0

    def test_buy_quote_currency(self):
        """Buy side quote notional is non-zero."""
        book = _mk_book()
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.0, 2.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.00],
            sizes=[1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        result = book.get_size_for_price_impact_bps(
            100.0, is_buy=True, is_base_currency=False
        )
        assert result > 0

    def test_sell_quote_currency(self):
        """Sell side quote notional is non-zero."""
        book = _mk_book()
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.0, 2.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.00],
            sizes=[1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        result = book.get_size_for_price_impact_bps(
            100.0, is_buy=False, is_base_currency=False
        )
        assert result > 0

    def test_large_impact_covers_all_levels(self, standard_book):
        """100% impact band covers all levels on the ask side."""
        result = standard_book.get_size_for_price_impact_bps(
            10000.0, is_buy=True, is_base_currency=True
        )
        assert result == pytest.approx(7.5)

    def test_small_impact_only_first_level(self, standard_book):
        """Tiny impact returns 0 or very small value, does not crash."""
        result = standard_book.get_size_for_price_impact_bps(
            0.01, is_buy=True, is_base_currency=True
        )
        assert result >= 0.0

    def test_empty_book_raises(self, empty_book):
        """Empty book raises RuntimeError."""
        with pytest.raises(RuntimeError):
            empty_book.get_size_for_price_impact_bps(
                100.0, is_buy=True, is_base_currency=True
            )

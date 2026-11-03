"""Python-specific BBO return-type contract.

Feed behavior lives in the shared reference scenario suite; ingestion and export
ownership contracts live in test_ingestion_contract.py.
"""

from __future__ import annotations

import pytest

from tests.orderbook.advanced.conftest import (
    PyOrderbookLevel,
    _make_levels,
    _mk_book,
)


class TestBBOReturnType:
    """The Python wrapper returns public Python level objects."""

    def test_get_bbo_returns_py_orderbook_levels(self):
        """Given populated book, When get_bbo called, Then returns PyOrderbookLevel instances."""
        book = _mk_book(num_levels=64)

        bids, _ = _make_levels(
            prices=[100.0, 99.99, 99.98],
            sizes=[1.0, 2.0, 3.0],
            norders=[1, 2, 3],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03],
            sizes=[1.5, 2.5, 3.5],
            norders=[1, 2, 3],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        best_bid, best_ask = book.get_bbo()

        assert isinstance(best_bid, PyOrderbookLevel)
        assert isinstance(best_ask, PyOrderbookLevel)
        assert best_bid.price == pytest.approx(100.0)
        assert best_bid.size == pytest.approx(1.0)
        assert best_bid.norders == 1
        assert best_ask.price == pytest.approx(100.01)
        assert best_ask.size == pytest.approx(1.5)
        assert best_ask.norders == 1

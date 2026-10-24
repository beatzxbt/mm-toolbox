"""End-to-end integration tests for the Python wrapper.

Layer 3 tests: realistic sequences through the Python wrapper covering
snapshot-delta-BBO workflows, numpy vs struct equivalence, clear/repopulate,
large capacity, crossed books, and BBO return types.
"""

from __future__ import annotations

import numpy as np
import pytest

from mm_toolbox.orderbook.advanced import OrderbookLevels
from tests.orderbook.advanced.conftest import (
    PyOrderbookLevel,
    _make_levels,
    _mk_book,
)


class TestEndToEnd:
    """Layer 3: Realistic sequences through the Python wrapper."""

    def test_snapshot_delta_bbo_sequence(self):
        """Given snapshot, delta, and BBO updates, When applied in sequence, Then mid price is valid."""
        book = _mk_book(num_levels=64)

        # Snapshot
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

        # Delta
        delta_asks, _ = _make_levels(
            prices=[100.01], sizes=[5.0], norders=[5], with_precision=True
        )
        empty_bids = OrderbookLevels.from_list([1.0], [0.0], [0])
        book.consume_deltas(delta_asks, empty_bids)

        # BBO
        book.consume_bbo_values(
            ask_price=100.015,
            ask_size=2.0,
            bid_price=100.005,
            bid_size=2.0,
        )

        assert book.get_mid_price() > 0

    def test_consume_bbo_values_updates_raw_exports(self):
        """Given scalar BBO updates, When consumed, Then exported raw levels update."""
        book = _mk_book(num_levels=64)

        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)
        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        book.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        book.consume_bbo_values(
            ask_price=100.01,
            ask_size=3.25,
            bid_price=100.0,
            bid_size=4.5,
            ask_norders=7,
            bid_norders=9,
        )

        bids = book.get_bids_numpy()
        asks = book.get_asks_numpy()
        assert bids["price"][0] == pytest.approx(100.0)
        assert asks["price"][0] == pytest.approx(100.01)
        assert bids["size"][0] == pytest.approx(4.5)
        assert asks["size"][0] == pytest.approx(3.25)
        assert bids["norders"][0] == 9
        assert asks["norders"][0] == 7

    def test_consume_bbo_values_and_level_struct_are_equivalent(self):
        """Given identical BBO inputs, When consumed as values or levels, Then books match."""
        values_book = _mk_book(num_levels=64)
        levels_book = _mk_book(num_levels=64)
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
        values_book.consume_snapshot(asks, bids)
        levels_book.consume_snapshot(asks, bids)

        values_book.consume_bbo_values(
            ask_price=100.005,
            ask_size=2.25,
            bid_price=99.995,
            bid_size=2.75,
            ask_norders=4,
            bid_norders=5,
        )
        levels_book.consume_bbo(
            PyOrderbookLevel(100.005, 2.25, 4),
            PyOrderbookLevel(99.995, 2.75, 5),
        )

        np.testing.assert_array_equal(
            np.asarray(values_book.get_asks_numpy()),
            np.asarray(levels_book.get_asks_numpy()),
        )
        np.testing.assert_array_equal(
            np.asarray(values_book.get_bids_numpy()),
            np.asarray(levels_book.get_bids_numpy()),
        )

    def test_crossed_bbo_values_rejected(self):
        """Given crossed scalar BBO values, When consumed, Then ValueError is raised."""
        book = _mk_book(num_levels=64)

        with pytest.raises(ValueError, match="Crossed BBO"):
            book.consume_bbo_values(
                ask_price=100.00,
                ask_size=1.0,
                bid_price=100.01,
                bid_size=1.0,
            )

    def test_numpy_vs_struct_equivalence(self):
        """Given same data via numpy and struct APIs, When consumed, Then produce identical mid prices."""
        ob_numpy = _mk_book(num_levels=64)
        ob_struct = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        ob_numpy.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bids, _ = _make_levels(
            prices=[100.0, 99.99],
            sizes=[1.0, 2.0],
            norders=[1, 1],
            with_precision=False,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.5, 2.5],
            norders=[1, 1],
            with_precision=False,
        )
        ob_struct.consume_snapshot(asks, bids)

        assert ob_numpy.get_mid_price() == ob_struct.get_mid_price()

    def test_clear_and_repopulate(self, standard_book):
        """Given a populated book, When cleared and repopulated, Then works correctly."""
        standard_book.clear()

        with pytest.raises(RuntimeError):
            standard_book.get_mid_price()

        bids, _ = _make_levels(
            prices=[200.0], sizes=[1.0], norders=[1], with_precision=True
        )
        asks, _ = _make_levels(
            prices=[200.01], sizes=[1.0], norders=[1], with_precision=True
        )
        standard_book.consume_snapshot(asks, bids)
        assert standard_book.get_mid_price() == pytest.approx(200.0)

    def test_large_capacity_book(self):
        """Given 1000-level book at capacity, When populated, Then mid price valid."""
        book = _mk_book(num_levels=1000)

        ask_prices = [100.0 + i * 0.01 for i in range(100)]
        bid_prices = [99.99 - i * 0.01 for i in range(100)]
        sizes = [1.0] * 100
        norders = [1] * 100

        asks = OrderbookLevels.from_list(ask_prices, sizes, norders)
        bids = OrderbookLevels.from_list(bid_prices, sizes, norders)
        book.consume_snapshot(asks, bids)

        assert book.get_mid_price() > 0

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

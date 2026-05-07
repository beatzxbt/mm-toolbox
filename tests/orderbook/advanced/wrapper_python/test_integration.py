"""End-to-end integration tests for the Python wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from mm_toolbox.orderbook.advanced import (
    OrderbookLevels,
)
from tests.orderbook.advanced.conftest import (
    TICK_SIZE,
    LOT_SIZE,
    _make_levels,
    _mk_book,
)


class TestEndToEnd:
    """Realistic sequences through the Python wrapper."""

    def test_snapshot_delta_bbo_sequence(self):
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
        empty_bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [1.0], [0.0], [0], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(delta_asks, empty_bids)

        # BBO
        from mm_toolbox.orderbook.advanced import OrderbookLevel

        new_ask = OrderbookLevel.with_ticks_and_lots(
            100.015, 2.0, TICK_SIZE, LOT_SIZE, 1
        )
        new_bid = OrderbookLevel.with_ticks_and_lots(
            100.005, 2.0, TICK_SIZE, LOT_SIZE, 1
        )
        book.consume_bbo(new_ask, new_bid)

        assert book.get_mid_price() > 0

    def test_numpy_vs_struct_equivalence(self):
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
        book = _mk_book(num_levels=1000)

        ask_prices = [100.0 + i * 0.01 for i in range(100)]
        bid_prices = [99.99 - i * 0.01 for i in range(100)]
        sizes = [1.0] * 100
        norders = [1] * 100

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            ask_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            bid_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )
        book.consume_snapshot(asks, bids)

        assert book.get_mid_price() > 0

    def test_crossed_book_handling(self):
        book = _mk_book(num_levels=64)

        # Crossed snapshot: bid > ask
        asks, _ = _make_levels(
            prices=[100.0], sizes=[1.0], norders=[1], with_precision=True
        )
        bids, _ = _make_levels(
            prices=[100.01], sizes=[1.0], norders=[1], with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Crossed books are preserved by snapshot
        spread = book.get_bbo_spread()
        assert spread < 0

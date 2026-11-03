"""Python wrapper constructor, numeric extremes, and raw-input boundaries.

Orderbook state transitions are covered by the shared reference scenario suite;
these checks retain wrapper-specific validation and precision boundaries.
"""

from __future__ import annotations

import pytest

from mm_toolbox.orderbook.advanced import (
    PyAdvancedOrderbook,
    OrderbookLevels,
    PyOrderbookSortedness,
)
from tests.orderbook.advanced.conftest import (
    TICK_SIZE,
    LOT_SIZE,
    _mk_book,
    _make_levels,
)


@pytest.mark.boundary
class TestEmptyOrderbookBoundaries:
    """Layer 2: Test empty orderbook operations and edge cases."""

    def test_operations_on_never_populated_orderbook(self):
        """Given never-populated book, When accessors called, Then empty arrays or appropriate errors."""
        book = _mk_book(num_levels=64)

        # get_*_numpy() should return empty arrays
        bids = book.get_bids_numpy()
        asks = book.get_asks_numpy()

        assert len(bids) == 0
        assert len(asks) == 0

        # get_mid_price() and get_bbo_spread() should raise on empty orderbook
        with pytest.raises(RuntimeError, match="Empty view"):
            book.get_mid_price()

        with pytest.raises(RuntimeError, match="Empty view"):
            book.get_bbo_spread()


@pytest.mark.boundary
class TestMaxCapacityBoundaries:
    """Capacity allocation and constructor limits beyond ordinary replay sizes."""

    @pytest.mark.slow
    def test_initialization_at_max_capacity(self):
        """Given num_levels=16777216 (ORDERBOOK_MAX_LEVELS), When created, Then succeeds."""
        # This test is slow due to large memory allocation
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=16777216,  # 2^24
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None

    def test_initialization_above_max_capacity(self):
        """Given num_levels > max, When created, Then raises ValueError."""
        # Note: Validation happens at OrderbookLadder level (ladder.pyx:58)
        # ORDERBOOK_MAX_LEVELS = 16777216 (2^24), so 16777217 should fail instantly
        with pytest.raises(ValueError, match="Invalid max_levels"):
            PyAdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=16777217,  # 2^24 + 1, exceeds max
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

    @pytest.mark.slow
    def test_snapshot_at_capacity_1000_exports_all_levels(self, max_capacity_orderbook):
        """A practical 1000-level capacity exports every retained row accurately."""
        book = max_capacity_orderbook

        # Create 1000 levels
        ask_prices = [100.0 + i * 0.01 for i in range(1000)]
        bid_prices = [99.99 - i * 0.01 for i in range(1000)]
        sizes = [1.0] * 1000
        norders = [1] * 1000

        asks = OrderbookLevels.from_list(ask_prices, sizes, norders)
        bids = OrderbookLevels.from_list(bid_prices, sizes, norders)

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        assert len(asks_arr) == 1000
        assert len(bids_arr) == 1000

        assert list(asks_arr["price"]) == pytest.approx(ask_prices, rel=0, abs=1e-12)
        assert list(bids_arr["price"]) == pytest.approx(bid_prices, rel=0, abs=1e-12)
        for levels in (asks_arr, bids_arr):
            assert list(levels["size"]) == sizes
            assert list(levels["norders"]) == norders


@pytest.mark.boundary
class TestPathologicalDataBoundaries:
    """Layer 2: Test pathological data scenarios."""

    def test_huge_spread_between_bbo(self, pathological_data):
        """Given bid=1.00 and ask=10000.00, When snapshot consumed, Then spread computed correctly."""
        book = _mk_book(num_levels=64)

        bid_price, ask_price = pathological_data["huge_spread"]

        asks = OrderbookLevels.from_list([ask_price], [1.0], [1])
        bids = OrderbookLevels.from_list([bid_price], [1.0], [1])

        book.consume_snapshot(asks, bids)

        spread = book.get_bbo_spread()
        assert spread == pytest.approx(ask_price - bid_price)

    def test_extreme_price_values(self, pathological_data):
        """Given large tick-representable prices, When consumed, Then handles correctly."""
        book = _mk_book(num_levels=64)

        extreme_high = pathological_data["extreme_price_high"]
        extreme_low = pathological_data["extreme_price_low"]

        asks = OrderbookLevels.from_list([extreme_high], [1.0], [1])
        bids = OrderbookLevels.from_list([extreme_low], [1.0], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        assert asks_arr["price"][0] == pytest.approx(extreme_high)
        assert bids_arr["price"][0] == pytest.approx(extreme_low)

    def test_extreme_size_values(self, pathological_data):
        """Given size=1e15, When consumed, Then handles correctly."""
        book = _mk_book(num_levels=64)

        extreme_size = pathological_data["extreme_size"]

        asks = OrderbookLevels.from_list([100.0], [extreme_size], [1])
        bids = OrderbookLevels.from_list([99.99], [extreme_size], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        assert asks_arr["size"][0] == pytest.approx(extreme_size)

    def test_extremely_small_tick_size(self):
        """Given tick_size=1e-10, When book created, Then handles precision."""
        book = PyAdvancedOrderbook(
            tick_size=1e-10,
            lot_size=LOT_SIZE,
            num_levels=64,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )

        asks = OrderbookLevels.from_list([100.0], [1.0], [1])
        bids = OrderbookLevels.from_list([99.99], [1.0], [1])

        book.consume_snapshot(asks, bids)

        assert book.get_asks_numpy()["price"][0] == pytest.approx(100.0)
        assert book.get_bids_numpy()["price"][0] == pytest.approx(99.99)
        assert book.get_bbo_spread() == pytest.approx(0.01, rel=0, abs=1e-12)

    def test_extremely_large_tick_size(self):
        """Given tick_size=1000.0, When book created, Then handles correctly."""
        book = PyAdvancedOrderbook(
            tick_size=1000.0,
            lot_size=LOT_SIZE,
            num_levels=64,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )

        asks = OrderbookLevels.from_list([100000.0], [1.0], [1])
        bids = OrderbookLevels.from_list([99000.0], [1.0], [1])

        book.consume_snapshot(asks, bids)

        assert book.get_asks_numpy()["price"][0] == 100000.0
        assert book.get_bids_numpy()["price"][0] == 99000.0
        assert book.get_bbo_spread() == 1000.0

    def test_out_of_order_snapshot_with_unknown_sortedness(self):
        """Given out-of-order snapshot with unknown sortedness, When consumed, Then raises."""
        book = _mk_book(num_levels=64)

        # Out of order asks
        asks = OrderbookLevels.from_list(
            [100.02, 100.0, 100.01],  # Wrong order
            [1.0, 1.0, 1.0],
            [1, 1, 1],
        )
        bids = OrderbookLevels.from_list(
            [99.97, 99.99, 99.98],  # Wrong order
            [1.0, 1.0, 1.0],
            [1, 1, 1],
        )

        with pytest.raises(ValueError, match="Unsorted orderbook levels"):
            book.consume_snapshot(asks, bids)


@pytest.mark.boundary
class TestPrecisionAndRoundingBoundaries:
    """Layer 2: Test precision and rounding edge cases."""

    def test_tick_rounding_near_boundaries(self):
        """Given price=100.005, When ingested, Then exported price is tick-rounded."""
        book = _mk_book(num_levels=64)
        asks = OrderbookLevels.from_list([100.005], [1.0], [1])
        bids = OrderbookLevels.from_list([99.99], [1.0], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        assert asks_arr["price"][0] == 100.0

    def test_lot_rounding_near_boundaries(self):
        """Given size=1.0005, When ingested, Then exported size is lot-rounded."""
        book = _mk_book(num_levels=64)
        asks = OrderbookLevels.from_list([100.01], [1.0005], [1])
        bids = OrderbookLevels.from_list([100.0], [1.0], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        assert asks_arr["size"][0] == 1.0


@pytest.mark.boundary
class TestLargeDeltaBatches:
    """Layer 2: Test large delta batches that stress the per-entry processing path."""

    def test_unsorted_large_mixed_batch_rejected(self):
        """Given unsorted mixed deltas, When consumed, Then raises."""
        book = _mk_book(num_levels=64)

        ask_prices = [100.0 + i * 0.01 for i in range(20)]
        bid_prices = [99.99 - i * 0.01 for i in range(20)]
        asks, _ = _make_levels(ask_prices, [1.0] * 20, with_precision=True)
        bids, _ = _make_levels(bid_prices, [1.0] * 20, with_precision=True)
        book.consume_snapshot(asks, bids)

        mix_ask_prices = []
        mix_ask_sizes = []
        for j in range(30):
            if j % 3 == 0:
                mix_ask_prices.append(100.0 + j * 0.005)
                mix_ask_sizes.append(2.0)
            elif j % 3 == 1:
                mix_ask_prices.append(100.0 + (j % 20) * 0.01)
                mix_ask_sizes.append(0.0)
            else:
                mix_ask_prices.append(100.0 + (j % 20) * 0.01)
                mix_ask_sizes.append(3.0)

        mix_bid_prices = []
        mix_bid_sizes = []
        for j in range(30):
            if j % 3 == 0:
                mix_bid_prices.append(99.99 - j * 0.005)
                mix_bid_sizes.append(2.0)
            elif j % 3 == 1:
                mix_bid_prices.append(99.99 - (j % 20) * 0.01)
                mix_bid_sizes.append(0.0)
            else:
                mix_bid_prices.append(99.99 - (j % 20) * 0.01)
                mix_bid_sizes.append(3.0)

        before = (book.get_asks_numpy().tolist(), book.get_bids_numpy().tolist())
        delta_asks = OrderbookLevels.from_list(mix_ask_prices, mix_ask_sizes)
        delta_bids = OrderbookLevels.from_list(mix_bid_prices, mix_bid_sizes)
        with pytest.raises(ValueError, match="Unsorted orderbook levels"):
            book.consume_deltas(delta_asks, delta_bids)

        assert (
            book.get_asks_numpy().tolist(),
            book.get_bids_numpy().tolist(),
        ) == before

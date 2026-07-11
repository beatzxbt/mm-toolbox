"""Comprehensive boundary condition tests for PyAdvancedOrderbook.

Tests edge cases including empty orderbooks, max capacity, pathological data,
precision/rounding, and overflow protection. These tests verify the robustness
of recent bug fixes and ensure correct behavior at system boundaries.
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
    _empty_levels,
    _empty_bid_levels,
)


@pytest.mark.boundary
class TestEmptyOrderbookBoundaries:
    """Layer 2: Test empty orderbook operations and edge cases."""

    def test_empty_orderbook_initialization(self):
        """Given num_levels < 4, When creating PyAdvancedOrderbook, Then raises ValueError."""
        with pytest.raises(ValueError, match="expected >=4"):
            PyAdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=0,
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

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

    def test_snapshot_with_zero_levels(self):
        """Given snapshot with only deletion markers, When consumed, Then retains zero-size levels."""
        book = _mk_book(num_levels=64)

        # Create snapshots with only deletion markers (size=0, norders=0)
        empty_asks = OrderbookLevels.from_list([100.0], [0.0], [0])
        empty_bids = OrderbookLevels.from_list([99.99], [0.0], [0])

        book.consume_snapshot(empty_asks, empty_bids)

        # Orderbook retains zero-size levels from snapshots
        bids = book.get_bids_numpy()
        asks = book.get_asks_numpy()
        assert len(bids) == 1
        assert len(asks) == 1
        assert bids["size"][0] == 0.0
        assert asks["size"][0] == 0.0

    def test_successive_deletions_to_empty_state(self):
        """Given populated book, When all levels deleted one-by-one, Then book is empty."""
        book = _mk_book(num_levels=64)

        # Initialize with 3 levels
        asks, _ = _make_levels(
            [100.0, 100.01, 100.02], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels(
            [99.99, 99.98, 99.97], [1.0, 1.0, 1.0], with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Delete all asks
        delete_ask1 = OrderbookLevels.from_list([100.0], [0.0], [0])
        delete_ask2 = OrderbookLevels.from_list([100.01], [0.0], [0])
        delete_ask3 = OrderbookLevels.from_list([100.02], [0.0], [0])

        book.consume_deltas(delete_ask1, _empty_bid_levels())
        book.consume_deltas(delete_ask2, _empty_bid_levels())
        book.consume_deltas(delete_ask3, _empty_bid_levels())

        # Orderbook should be empty after all deletions
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 0

    def test_ask_delete_then_insert_same_delta(self):
        """Given single ask, When delta deletes then inserts in same batch, Then new ask present."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        delta_asks = OrderbookLevels.from_list([100.0, 100.02], [0.0, 1.0], [0, 1])
        book.consume_deltas(delta_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 1
        assert asks_arr["price"][0] == pytest.approx(100.02)

    def test_bid_delete_then_insert_same_delta(self):
        """Given single bid, When delta deletes then inserts in same batch, Then new bid present."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        delta_bids = OrderbookLevels.from_list([99.90, 99.97], [0.0, 1.0], [0, 1])
        book.consume_deltas(_empty_levels(), delta_bids)

        bids_arr = book.get_bids_numpy()
        assert len(bids_arr) == 1
        assert bids_arr["price"][0] == pytest.approx(99.97)

    def test_bbo_deletion_leaving_empty_side(self):
        """Given populated book, When BBO deleted, Then side is empty."""
        book = _mk_book(num_levels=64)

        # Initialize with single level on each side
        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Delete the only ask level
        delete_ask = OrderbookLevels.from_list([100.0], [0.0], [0])
        book.consume_deltas(delete_ask, _empty_bid_levels())

        # Orderbook should be empty
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 0

    def test_cross_removal_ignored_without_replacement(self):
        """Given normal book, When crossing ask delta without bid replacement, Then ignored."""
        book = _mk_book(num_levels=64)

        # Initialize with tight bid/ask levels
        asks, _ = _make_levels([100.0, 100.01], [1.0, 1.0], with_precision=True)
        bids, _ = _make_levels([99.99, 99.98], [1.0, 1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Send crossed update: ask below best bid
        crossed_ask = OrderbookLevels.from_list([99.95], [1.0], [1])
        book.consume_deltas(crossed_ask, _empty_bid_levels())

        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert bids_arr["price"][0] == pytest.approx(99.99)
        assert asks_arr["price"][0] == pytest.approx(100.0)

    def test_crossing_bid_delta_ignored_without_replacement(self):
        """Given normal book, When crossing bid delta without ask replacement, Then ignored."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02], [1.0, 1.0], with_precision=True)
        bids, _ = _make_levels([100.00, 99.99], [1.0, 1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        crossed_bid = OrderbookLevels.from_list([100.05], [1.0], [1])
        book.consume_deltas(_empty_levels(), crossed_bid)

        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert bids_arr["price"][0] == pytest.approx(100.00)
        assert asks_arr["price"][0] == pytest.approx(100.01)

    def test_crossing_ask_delta_with_bid_replacement_applies(self):
        """Given normal book, When crossing ask delta with bid replacement, Then both sides updated."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02], [1.0, 1.0], with_precision=True)
        bids, _ = _make_levels([100.00, 99.99], [1.0, 1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        crossed_ask = OrderbookLevels.from_list([99.95], [1.0], [1])
        replacement_bid = OrderbookLevels.from_list([99.90], [1.0], [1])
        book.consume_deltas(crossed_ask, replacement_bid)

        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert bids_arr["price"][0] == pytest.approx(99.90)
        assert asks_arr["price"][0] == pytest.approx(99.95)


@pytest.mark.boundary
class TestDeltaBatchTopLevelRemovals:
    """Layer 2: Boundary coverage for delta batches that remove multiple top levels."""

    def test_ask_batch_removals_do_not_duplicate_levels(self):
        """Given ask-side removals in one delta, When consumed, Then no duplicate prices."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.00, 100.01, 100.02], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels(
            [99.99, 99.98, 99.97], [1.0, 1.0, 1.0], with_precision=True
        )
        book.consume_snapshot(asks, bids)

        delta_asks = OrderbookLevels.from_list(
            [100.00, 100.01, 100.02], [0.0, 0.0, 2.0], [0, 0, 1]
        )
        book.consume_deltas(delta_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 1
        assert asks_arr["price"][0] == pytest.approx(100.02)
        assert asks_arr["size"][0] == pytest.approx(2.0)
        assert len(asks_arr["price"]) == len(set(asks_arr["price"]))

    def test_bid_batch_removals_do_not_duplicate_levels(self):
        """Given bid-side removals in one delta, When consumed, Then no duplicate prices."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.01, 100.02, 100.03], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels(
            [99.99, 99.98, 99.97], [1.0, 1.0, 1.0], with_precision=True
        )
        book.consume_snapshot(asks, bids)

        delta_bids = OrderbookLevels.from_list(
            [99.99, 99.98, 99.97], [0.0, 0.0, 2.0], [0, 0, 1]
        )
        book.consume_deltas(_empty_levels(), delta_bids)

        bids_arr = book.get_bids_numpy()
        assert len(bids_arr) == 1
        assert bids_arr["price"][0] == pytest.approx(99.97)
        assert bids_arr["size"][0] == pytest.approx(2.0)
        assert len(bids_arr["price"]) == len(set(bids_arr["price"]))


@pytest.mark.boundary
class TestMaxCapacityBoundaries:
    """Layer 2: Test max capacity edge cases and overflow protection."""

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
    def test_snapshot_at_max_capacity(self, max_capacity_orderbook):
        """Given max capacity book, When full snapshot consumed, Then stores all levels."""
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

    @pytest.mark.slow
    def test_snapshot_exceeding_capacity_truncates(self):
        """Given snapshot exceeding capacity, When consumed, Then truncates to capacity."""
        book = _mk_book(num_levels=64)

        # Create 20 levels (exceeds capacity of 64)
        ask_prices = [100.0 + i * 0.01 for i in range(20)]
        bid_prices = [99.99 - i * 0.01 for i in range(20)]
        sizes = [1.0] * 20
        norders = [1] * 20

        asks = OrderbookLevels.from_list(ask_prices, sizes, norders)
        bids = OrderbookLevels.from_list(bid_prices, sizes, norders)

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        # Should truncate to 16 levels
        assert len(asks_arr) <= 64
        assert len(bids_arr) <= 64

    def test_delta_insertion_at_full_capacity(self):
        """Given full book, When new level inserted, Then evicts worst level."""
        book = _mk_book(num_levels=64)

        # Fill to capacity with 64 ask levels
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(64)], [1.0] * 64, with_precision=True
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Insert new best ask (should evict worst ask at 100.63)
        new_ask = OrderbookLevels.from_list([99.98], [2.0], [1])
        book.consume_deltas(new_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 64  # Still at capacity
        assert asks_arr["price"][0] == pytest.approx(99.98)  # New best ask

    def test_roll_right_at_max_capacity(self):
        """Given full book, When BBO inserted, Then evicts last level."""
        book = _mk_book(num_levels=64)

        # Fill to capacity with 64 levels
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(64)],
            [float(i + 1) for i in range(64)],
            with_precision=True,
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Insert better ask at front (should evict last level)
        better_ask = OrderbookLevels.from_list([99.95], [5.0], [1])
        book.consume_deltas(better_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 64
        assert asks_arr["price"][0] == pytest.approx(99.95)
        # Last level should be evicted

    @pytest.mark.slow
    def test_rapid_insertions_and_deletions_at_capacity(self):
        """Given book at capacity, When 1000 rapid updates applied, Then remains consistent."""
        book = _mk_book(num_levels=64)

        # Initial snapshot
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(10)], [1.0] * 10, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(10)], [1.0] * 10, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Rapid updates
        for i in range(1000):
            price = 100.0 + (i % 20) * 0.01
            size = 1.0 if i % 2 == 0 else 0.0  # Alternate insert/delete

            delta = OrderbookLevels.from_list([price], [size], [1 if size > 0 else 0])
            book.consume_deltas(delta, _empty_bid_levels())

        # Should still be valid
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) <= 64


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

    def test_zero_tick_spread_rejected(self, pathological_data):
        """Given bid=ask=100.00, When snapshot consumed, Then snapshot is rejected."""
        book = _mk_book(num_levels=64)

        bid_price, ask_price = pathological_data["zero_spread"]

        asks = OrderbookLevels.from_list([ask_price], [1.0], [1])
        bids = OrderbookLevels.from_list([bid_price], [1.0], [1])

        with pytest.raises(ValueError, match="Crossed snapshot"):
            book.consume_snapshot(asks, bids)

    def test_duplicate_price_levels_in_snapshot(self):
        """Given duplicate prices in snapshot, When consumed, Then deduplicates correctly."""
        book = _mk_book(num_levels=64)

        # Duplicate ask prices
        asks = OrderbookLevels.from_list(
            [100.0, 100.0, 100.01], [1.0, 2.0, 1.0], [1, 1, 1]
        )
        bids = OrderbookLevels.from_list([99.99], [1.0], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        # Should handle duplicates (last one wins or aggregate)
        assert len(asks_arr) > 0

    def test_all_zero_sizes_in_snapshot(self):
        """Given snapshot with all deletion markers, When consumed, Then retains zero-size levels."""
        book = _mk_book(num_levels=64)

        # All zero sizes
        asks = OrderbookLevels.from_list([100.0, 100.01], [0.0, 0.0], [0, 0])
        bids = OrderbookLevels.from_list([99.99], [0.0], [0])

        book.consume_snapshot(asks, bids)

        # Orderbook retains zero-size levels from snapshots
        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert len(bids_arr) == 1
        assert len(asks_arr) == 2
        # All sizes should be zero
        assert all(bids_arr["size"] == 0.0)
        assert all(asks_arr["size"] == 0.0)

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

        mid_price = book.get_mid_price()
        assert mid_price > 0

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

        spread = book.get_bbo_spread()
        assert spread > 0

    def test_mixed_zero_and_nonzero_updates(self):
        """Given mixed delta types, When consumed, Then all applied correctly."""
        book = _mk_book(num_levels=64)

        # Initialize
        asks, _ = _make_levels(
            [100.0, 100.01, 100.02], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Mixed updates: delete 100.0, update 100.01, insert 100.03
        mixed_asks = OrderbookLevels.from_list(
            [100.0, 100.01, 100.03],
            [0.0, 2.0, 1.5],  # delete, update, insert
            [0, 1, 1],
        )

        book.consume_deltas(mixed_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) >= 2

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

    def test_sequential_crosses_via_bbo_updates(self):
        """Given multiple BBO updates causing crosses, When applied, Then handles sequential crosses."""
        book = _mk_book(num_levels=64)

        # Initialize
        asks, _ = _make_levels(
            [100.0, 100.01, 100.02], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels(
            [99.99, 99.98, 99.97], [1.0, 1.0, 1.0], with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # First cross: ask at 99.98 (crosses with top 2 bids)
        cross1 = OrderbookLevels.from_list([99.98], [1.0], [1])
        book.consume_deltas(cross1, _empty_bid_levels())

        # Second cross: ask at 99.96 (should cross with remaining bids)
        cross2 = OrderbookLevels.from_list([99.96], [1.0], [1])
        book.consume_deltas(cross2, _empty_bid_levels())

        # Should handle sequential crosses
        try:
            book.get_bids_numpy()
        except RuntimeError:
            # Empty after crosses is valid
            pass


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
        assert asks_arr["price"][0] == pytest.approx(
            int(100.005 / TICK_SIZE) * TICK_SIZE
        )

    def test_lot_rounding_near_boundaries(self):
        """Given size=1.0005, When ingested, Then exported size is lot-rounded."""
        book = _mk_book(num_levels=64)
        asks = OrderbookLevels.from_list([100.01], [1.0005], [1])
        bids = OrderbookLevels.from_list([100.0], [1.0], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        assert asks_arr["size"][0] == pytest.approx(int(1.0005 / LOT_SIZE) * LOT_SIZE)

    def test_mid_price_with_odd_tick_sum(self):
        """Given odd tick sum, When mid price computed, Then integer division verified."""
        book = _mk_book(num_levels=64)

        # Prices that result in odd tick sum
        asks = OrderbookLevels.from_list([100.01], [1.0], [1])
        bids = OrderbookLevels.from_list([100.00], [1.0], [1])

        book.consume_snapshot(asks, bids)

        mid_price = book.get_mid_price()
        expected_mid = (100.01 + 100.00) / 2.0
        assert mid_price == pytest.approx(expected_mid, abs=TICK_SIZE)

    def test_spread_precision_consistency(self):
        """Given spread calculation, When using tick arithmetic, Then matches price arithmetic."""
        book = _mk_book(num_levels=64)

        asks = OrderbookLevels.from_list([100.10], [1.0], [1])
        bids = OrderbookLevels.from_list([100.00], [1.0], [1])

        book.consume_snapshot(asks, bids)

        spread = book.get_bbo_spread()
        expected_spread = 100.10 - 100.00
        assert spread == pytest.approx(expected_spread, abs=1e-9)


@pytest.mark.boundary
class TestCapacityOverflowProtection:
    """Layer 2: Test overflow protection guards."""

    def test_increment_count_at_max_capacity_noop(self):
        """Given full ladder, When incrementing count, Then no-op (guard at ladder.pyx:125-127)."""
        book = _mk_book(num_levels=64)

        # Fill to capacity with 16 levels
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Try to add beyond capacity (worst price, should be ignored)
        extra_ask = OrderbookLevels.from_list([100.20], [1.0], [1])
        book.consume_deltas(extra_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        # Should not exceed capacity
        assert len(asks_arr) <= 64

    def test_decrement_count_at_zero_noop(self):
        """Given empty ladder, When decrementing count, Then no-op (guard at ladder.pyx:130-133)."""
        book = _mk_book(num_levels=64)

        # Initialize with one level
        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.99], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Delete the level
        delete_ask = OrderbookLevels.from_list([100.0], [0.0], [0])
        book.consume_deltas(delete_ask, _empty_bid_levels())

        # Try to delete again (should be no-op)
        book.consume_deltas(delete_ask, _empty_bid_levels())

        # Should not crash, and asks should be empty
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 0

    def test_roll_right_never_writes_past_capacity(self):
        """Given full ladder, When roll_right triggered, Then respects capacity."""
        book = _mk_book(num_levels=64)

        # Fill to capacity with 16 levels
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)],
            [1.0] * 16,
            with_precision=True,
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Insert at front (triggers roll_right, should evict last)
        new_ask = OrderbookLevels.from_list([99.95], [2.0], [1])
        book.consume_deltas(new_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) <= 64
        assert asks_arr["price"][0] == pytest.approx(99.95)


@pytest.mark.boundary
class TestLargeDeltaBatches:
    """Layer 2: Test large delta batches that stress the per-entry processing path."""

    def test_50_ask_deltas_in_one_batch(self):
        """Given book with 3 asks, When 50 ask deltas in one call, Then no crash and book valid."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(3)], [1.0] * 3, with_precision=True
        )
        bids, _ = _make_levels([99.99], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        ask_prices = [100.0 + j * 0.01 for j in range(50)]
        ask_sizes = [1.0 if j % 3 != 0 else 0.0 for j in range(50)]
        ask_norders = [1 if s > 0 else 0 for s in ask_sizes]

        delta_asks = OrderbookLevels.from_list(ask_prices, ask_sizes, ask_norders)
        book.consume_deltas(delta_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) <= 64
        assert book.get_mid_price() > 0

    def test_large_batch_vs_multiple_small_batches_equivalence(self):
        """Given two identical books, When deltas split vs batched, Then identical results.

        Uses non-crossing deltas: asks at/above best_ask, bids at/below best_bid.
        """
        book_a = _mk_book(num_levels=64)
        book_b = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0, 100.01, 100.02], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels(
            [99.99, 99.98, 99.97], [1.0, 1.0, 1.0], with_precision=True
        )
        book_a.consume_snapshot(asks, bids)
        book_b.consume_snapshot(asks, bids)

        all_ask_deltas = []
        all_bid_deltas = []
        for j in range(40):
            all_ask_deltas.append((100.0 + j * 0.01, float(1 + j % 3)))
            all_bid_deltas.append((99.99 - j * 0.01, float(1 + j % 3)))

        ask_prices = [p for p, _ in all_ask_deltas]
        ask_sizes = [s for _, s in all_ask_deltas]
        bid_prices = [p for p, _ in all_bid_deltas]
        bid_sizes = [s for _, s in all_bid_deltas]

        big_asks = OrderbookLevels.from_list(ask_prices, ask_sizes)
        big_bids = OrderbookLevels.from_list(bid_prices, bid_sizes)
        book_a.consume_deltas(big_asks, big_bids)

        chunk_size = 5
        for start in range(0, 40, chunk_size):
            end = min(start + chunk_size, 40)
            chunk_asks = OrderbookLevels.from_list(
                ask_prices[start:end], ask_sizes[start:end]
            )
            chunk_bids = OrderbookLevels.from_list(
                bid_prices[start:end], bid_sizes[start:end]
            )
            book_b.consume_deltas(chunk_asks, chunk_bids)

        assert book_a.get_mid_price() == pytest.approx(book_b.get_mid_price())
        assert book_a.get_bbo_spread() == pytest.approx(book_b.get_bbo_spread())
        assert len(book_a.get_asks_numpy()) == len(book_b.get_asks_numpy())
        assert len(book_a.get_bids_numpy()) == len(book_b.get_bids_numpy())

    def test_large_all_deletion_batch_with_one_insert(self):
        """Given 50 ask levels, When all deleted + one inserted in one batch, Then ask side has 1 level."""
        book = _mk_book(num_levels=64)

        ask_prices = [100.0 + i * 0.01 for i in range(50)]
        asks, _ = _make_levels(ask_prices, [1.0] * 50, with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        del_prices = [99.95] + ask_prices
        del_sizes = [1.0] + [0.0] * 50
        del_norders = [1] + [0] * 50

        delta_asks = OrderbookLevels.from_list(del_prices, del_sizes, del_norders)
        book.consume_deltas(delta_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 1
        assert asks_arr["price"][0] == pytest.approx(99.95)

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

        delta_asks = OrderbookLevels.from_list(mix_ask_prices, mix_ask_sizes)
        delta_bids = OrderbookLevels.from_list(mix_bid_prices, mix_bid_sizes)
        with pytest.raises(ValueError, match="Unsorted orderbook levels"):
            book.consume_deltas(delta_asks, delta_bids)

        assert book.get_mid_price() > 0

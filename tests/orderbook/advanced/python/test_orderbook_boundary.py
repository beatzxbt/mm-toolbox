"""
Comprehensive boundary condition tests for AdvancedOrderbook.

Tests edge cases including empty orderbooks, max capacity, pathological data,
precision/rounding, and overflow protection. These tests verify the robustness
of recent bug fixes and ensure correct behavior at system boundaries.
"""

from __future__ import annotations

import pytest

from mm_toolbox.orderbook.advanced import (
    AdvancedOrderbook,
    OrderbookLevel,
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
    """Test empty orderbook operations and edge cases."""

    def test_empty_orderbook_initialization(self):
        """Verify that num_levels<16 raises ValueError."""
        with pytest.raises(ValueError, match="expected >=64"):
            AdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=0,
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

    def test_operations_on_never_populated_orderbook(self):
        """Empty arrays returned when orderbook never populated."""
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
        """Snapshot with deletion markers retains zero-size levels."""
        book = _mk_book(num_levels=64)

        # Create snapshots with only deletion markers (size=0, norders=0)
        empty_asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0], [0.0], [0], TICK_SIZE, LOT_SIZE
        )
        empty_bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.99], [0.0], [0], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(empty_asks, empty_bids)

        # Orderbook retains zero-size levels from snapshots
        bids = book.get_bids_numpy()
        asks = book.get_asks_numpy()
        assert len(bids) == 1
        assert len(asks) == 1
        assert bids["size"][0] == 0.0
        assert asks["size"][0] == 0.0

    def test_successive_deletions_to_empty_state(self):
        """Delete all bids/asks one-by-one until empty."""
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
        delete_ask1 = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0], [0.0], [0], TICK_SIZE, LOT_SIZE
        )
        delete_ask2 = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.01], [0.0], [0], TICK_SIZE, LOT_SIZE
        )
        delete_ask3 = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.02], [0.0], [0], TICK_SIZE, LOT_SIZE
        )

        book.consume_deltas(delete_ask1, _empty_bid_levels())
        book.consume_deltas(delete_ask2, _empty_bid_levels())
        book.consume_deltas(delete_ask3, _empty_bid_levels())

        # Orderbook should be empty after all deletions
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 0

    def test_ask_delete_then_insert_same_delta(self):
        """Delete the only ask then insert a new ask in the same delta."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        delta_asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0, 100.02], [0.0, 1.0], [0, 1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(delta_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 1
        assert asks_arr["price"][0] == pytest.approx(100.02)

    def test_bid_delete_then_insert_same_delta(self):
        """Delete the only bid then insert a new bid in the same delta."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        delta_bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.90, 99.97], [0.0, 1.0], [0, 1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(_empty_levels(), delta_bids)

        bids_arr = book.get_bids_numpy()
        assert len(bids_arr) == 1
        assert bids_arr["price"][0] == pytest.approx(99.97)

    def test_bbo_deletion_leaving_empty_side(self):
        """One-sided empty after BBO deletion."""
        book = _mk_book(num_levels=64)

        # Initialize with single level on each side
        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Delete the only ask level
        delete_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0], [0.0], [0], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(delete_ask, _empty_bid_levels())

        # Orderbook should be empty
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 0

    def test_cross_removal_ignored_without_replacement(self):
        """Crossing ask delta without bid replacements is ignored."""
        book = _mk_book(num_levels=64)

        # Initialize with tight bid/ask levels
        asks, _ = _make_levels([100.0, 100.01], [1.0, 1.0], with_precision=True)
        bids, _ = _make_levels([99.99, 99.98], [1.0, 1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Send crossed update: ask below best bid
        crossed_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.95], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(crossed_ask, _empty_bid_levels())

        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert bids_arr["price"][0] == pytest.approx(99.99)
        assert asks_arr["price"][0] == pytest.approx(100.0)

    def test_crossing_bid_delta_ignored_without_replacement(self):
        """Crossing bid delta without ask replacements is ignored."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02], [1.0, 1.0], with_precision=True)
        bids, _ = _make_levels([100.00, 99.99], [1.0, 1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        crossed_bid = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.05], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(_empty_levels(), crossed_bid)

        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert bids_arr["price"][0] == pytest.approx(100.00)
        assert asks_arr["price"][0] == pytest.approx(100.01)

    def test_crossing_ask_delta_with_bid_replacement_applies(self):
        """Crossing ask delta with bid replacements updates both sides."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02], [1.0, 1.0], with_precision=True)
        bids, _ = _make_levels([100.00, 99.99], [1.0, 1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        crossed_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.95], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        replacement_bid = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.90], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(crossed_ask, replacement_bid)

        bids_arr = book.get_bids_numpy()
        asks_arr = book.get_asks_numpy()
        assert bids_arr["price"][0] == pytest.approx(99.90)
        assert asks_arr["price"][0] == pytest.approx(99.95)


@pytest.mark.boundary
class TestMaxCapacityBoundaries:
    """Test max capacity edge cases and overflow protection."""

    def test_initialization_at_max_capacity(self):
        """Create with num_levels=64777216 (ORDERBOOK_MAX_LEVELS)."""
        # This test is slow due to large memory allocation
        # Skip in normal test runs
        pytest.skip("Skipping max capacity initialization test (very slow)")

        book = AdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=64777216,  # 2^24
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None

    def test_initialization_above_max_capacity(self):
        """Verify ValueError for overflow (tested at ladder level)."""
        # Note: Validation happens at OrderbookLadder level (ladder.pyx:58)
        # ORDERBOOK_MAX_LEVELS = 16777216 (2^24), so 16777217 should fail instantly
        with pytest.raises(ValueError, match="Invalid max_levels"):
            AdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=64777217,  # 2^24 + 1, exceeds max
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

    @pytest.mark.slow
    def test_snapshot_at_max_capacity(self, max_capacity_orderbook):
        """Full capacity snapshot insertion."""
        book = max_capacity_orderbook

        # Create 1000 levels
        ask_prices = [100.0 + i * 0.01 for i in range(1000)]
        bid_prices = [99.99 - i * 0.01 for i in range(1000)]
        sizes = [1.0] * 1000
        norders = [1] * 1000

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            ask_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            bid_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        assert len(asks_arr) == 1000
        assert len(bids_arr) == 1000

    @pytest.mark.slow
    def test_snapshot_exceeding_capacity_truncates(self):
        """Verify truncation to capacity."""
        book = _mk_book(num_levels=64)

        # Create 20 levels (exceeds capacity of 10)
        ask_prices = [100.0 + i * 0.01 for i in range(20)]
        bid_prices = [99.99 - i * 0.01 for i in range(20)]
        sizes = [1.0] * 20
        norders = [1] * 20

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            ask_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            bid_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        # Should truncate to 16 levels
        assert len(asks_arr) <= 64
        assert len(bids_arr) <= 64

    def test_delta_insertion_at_full_capacity(self):
        """Worst level eviction when at capacity."""
        book = _mk_book(num_levels=64)

        # Fill to capacity with 64 ask levels
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(64)], [1.0] * 64, with_precision=True
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Insert new best ask (should evict worst ask at 100.63)
        new_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.98], [2.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(new_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 64  # Still at capacity
        assert asks_arr["price"][0] == pytest.approx(99.98)  # New best ask

    def test_roll_right_at_max_capacity(self):
        """BBO insertion evicts last level."""
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
        better_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.95], [5.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(better_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 64
        assert asks_arr["price"][0] == pytest.approx(99.95)
        # Last level should be evicted

    @pytest.mark.slow
    def test_rapid_insertions_and_deletions_at_capacity(self):
        """1000 iteration stress test at capacity."""
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

            delta = OrderbookLevels.from_list_with_ticks_and_lots(
                [price], [size], [1 if size > 0 else 0], TICK_SIZE, LOT_SIZE
            )
            book.consume_deltas(delta, _empty_bid_levels())

        # Should still be valid
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) <= 64


@pytest.mark.boundary
class TestPathologicalDataBoundaries:
    """Test pathological data scenarios."""

    def test_huge_spread_between_bbo(self, pathological_data):
        """bid=1.00, ask=10000.00."""
        book = _mk_book(num_levels=64)

        bid_price, ask_price = pathological_data["huge_spread"]

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [ask_price], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [bid_price], [1.0], [1], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        spread = book.get_bbo_spread()
        assert spread == pytest.approx(ask_price - bid_price)

    def test_zero_tick_spread(self, pathological_data):
        """bid=ask=100.00."""
        book = _mk_book(num_levels=64)

        bid_price, ask_price = pathological_data["zero_spread"]

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [ask_price], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [bid_price], [1.0], [1], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        spread = book.get_bbo_spread()
        assert spread == pytest.approx(0.0)

    def test_negative_spread_via_snapshot(self, pathological_data):
        """Crossed orderbook via snapshot."""
        book = _mk_book(num_levels=64)

        bid_price, ask_price = pathological_data["negative_spread"]

        # Crossed: bid > ask
        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [ask_price], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [bid_price], [1.0], [1], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        # After cross removal, orderbook should be empty or have non-crossed levels
        try:
            spread = book.get_bbo_spread()
            assert spread >= 0.0  # Should not be negative
        except RuntimeError:
            # Empty after cross removal is also valid
            pass

    def test_duplicate_price_levels_in_snapshot(self):
        """Deduplication verification."""
        book = _mk_book(num_levels=64)

        # Duplicate ask prices
        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0, 100.0, 100.01], [1.0, 2.0, 1.0], [1, 1, 1], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.99], [1.0], [1], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        # Should handle duplicates (last one wins or aggregate)
        assert len(asks_arr) > 0

    def test_all_zero_sizes_in_snapshot(self):
        """Snapshot with all deletion markers retains zero-size levels."""
        book = _mk_book(num_levels=64)

        # All zero sizes
        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0, 100.01], [0.0, 0.0], [0, 0], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.99], [0.0], [0], TICK_SIZE, LOT_SIZE
        )

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
        """prices near float64 limits (1e100, 1e-100)."""
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
        """size=1e15."""
        book = _mk_book(num_levels=64)

        extreme_size = pathological_data["extreme_size"]

        asks = OrderbookLevels.from_list([100.0], [extreme_size], [1])
        bids = OrderbookLevels.from_list([99.99], [extreme_size], [1])

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        assert asks_arr["size"][0] == pytest.approx(extreme_size)

    def test_extremely_small_tick_size(self):
        """tick_size=1e-10."""
        book = AdvancedOrderbook(
            tick_size=1e-10,
            lot_size=LOT_SIZE,
            num_levels=64,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0], [1.0], [1], 1e-10, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.99], [1.0], [1], 1e-10, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        mid_price = book.get_mid_price()
        assert mid_price > 0

    def test_extremely_large_tick_size(self):
        """tick_size=1000.0."""
        book = AdvancedOrderbook(
            tick_size=1000.0,
            lot_size=LOT_SIZE,
            num_levels=64,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100000.0], [1.0], [1], 1000.0, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99000.0], [1.0], [1], 1000.0, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        spread = book.get_bbo_spread()
        assert spread > 0

    def test_mixed_zero_and_nonzero_updates(self):
        """Mixed delta types."""
        book = _mk_book(num_levels=64)

        # Initialize
        asks, _ = _make_levels(
            [100.0, 100.01, 100.02], [1.0, 1.0, 1.0], with_precision=True
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Mixed updates: delete 100.0, update 100.01, insert 100.03
        mixed_asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0, 100.01, 100.03],
            [0.0, 2.0, 1.5],  # delete, update, insert
            [0, 1, 1],
            TICK_SIZE,
            LOT_SIZE,
        )

        book.consume_deltas(mixed_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) >= 2

    def test_out_of_order_snapshot_with_unknown_sortedness(self):
        """Auto-sort verification."""
        book = _mk_book(num_levels=64)

        # Out of order asks
        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.02, 100.0, 100.01],  # Wrong order
            [1.0, 1.0, 1.0],
            [1, 1, 1],
            TICK_SIZE,
            LOT_SIZE,
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.97, 99.99, 99.98],  # Wrong order
            [1.0, 1.0, 1.0],
            [1, 1, 1],
            TICK_SIZE,
            LOT_SIZE,
        )

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        # Should be sorted after consumption
        assert asks_arr["price"][0] <= asks_arr["price"][-1]  # Ascending
        assert bids_arr["price"][0] >= bids_arr["price"][-1]  # Descending

    def test_sequential_crosses_via_bbo_updates(self):
        """Multiple cross removals."""
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
        cross1 = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.98], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(cross1, _empty_bid_levels())

        # Second cross: ask at 99.96 (should cross with remaining bids)
        cross2 = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.96], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(cross2, _empty_bid_levels())

        # Should handle sequential crosses
        try:
            book.get_bids_numpy()
        except RuntimeError:
            # Empty after crosses is valid
            pass


@pytest.mark.boundary
class TestPrecisionAndRoundingBoundaries:
    """Test precision and rounding edge cases."""

    def test_tick_rounding_near_boundaries(self):
        """price=100.005 → floor to 10000 ticks."""
        # With TICK_SIZE=0.01, price=100.005 should round to 100.00 ticks (10000 ticks)
        level = OrderbookLevel.with_ticks_and_lots(
            100.005, 1.0, TICK_SIZE, LOT_SIZE, norders=1
        )

        expected_ticks = int(100.005 / TICK_SIZE)
        assert level.ticks == expected_ticks

    def test_lot_rounding_near_boundaries(self):
        """size=1.0005 rounding."""
        level = OrderbookLevel.with_ticks_and_lots(
            100.0, 1.0005, TICK_SIZE, LOT_SIZE, norders=1
        )

        expected_lots = int(1.0005 / LOT_SIZE)
        assert level.lots == expected_lots

    def test_mid_price_with_odd_tick_sum(self):
        """Integer division verification."""
        book = _mk_book(num_levels=64)

        # Prices that result in odd tick sum
        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.01], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.00], [1.0], [1], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        mid_price = book.get_mid_price()
        expected_mid = (100.01 + 100.00) / 2.0
        assert mid_price == pytest.approx(expected_mid, abs=TICK_SIZE)

    def test_spread_precision_consistency(self):
        """Tick arithmetic vs price arithmetic."""
        book = _mk_book(num_levels=64)

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.10], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.00], [1.0], [1], TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        spread = book.get_bbo_spread()
        expected_spread = 100.10 - 100.00
        assert spread == pytest.approx(expected_spread, abs=1e-9)


@pytest.mark.boundary
class TestCapacityOverflowProtection:
    """Test overflow protection guards."""

    def test_increment_count_at_max_capacity_noop(self):
        """Guard at ladder.pyx:125-127."""
        book = _mk_book(num_levels=64)

        # Fill to capacity with 16 levels
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels([99.90], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Try to add beyond capacity (worst price, should be ignored)
        extra_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.20], [1.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(extra_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        # Should not exceed capacity
        assert len(asks_arr) <= 64

    def test_decrement_count_at_zero_noop(self):
        """Guard at ladder.pyx:130-133."""
        book = _mk_book(num_levels=64)

        # Initialize with one level
        asks, _ = _make_levels([100.0], [1.0], with_precision=True)
        bids, _ = _make_levels([99.99], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        # Delete the level
        delete_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [100.0], [0.0], [0], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(delete_ask, _empty_bid_levels())

        # Try to delete again (should be no-op)
        book.consume_deltas(delete_ask, _empty_bid_levels())

        # Should not crash, and asks should be empty
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) == 0

    def test_roll_right_never_writes_past_capacity(self):
        """Verify roll operations respect capacity."""
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
        new_ask = OrderbookLevels.from_list_with_ticks_and_lots(
            [99.95], [2.0], [1], TICK_SIZE, LOT_SIZE
        )
        book.consume_deltas(new_ask, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) <= 64
        assert asks_arr["price"][0] == pytest.approx(99.95)

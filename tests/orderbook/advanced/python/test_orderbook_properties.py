"""
Property-based and invariant tests for AdvancedOrderbook.

Tests that certain properties and invariants hold across a wide range of inputs
and sequences of operations. Uses pytest parametrization and randomized testing
instead of Hypothesis to avoid external dependencies.
"""

from __future__ import annotations

import random
import pytest
import numpy as np

from mm_toolbox.orderbook.advanced import (
    OrderbookLevels,
)
from tests.orderbook.advanced.conftest import (
    TICK_SIZE,
    LOT_SIZE,
    _mk_book,
    _empty_bid_levels,
)


def _generate_valid_snapshot(
    num_bids: int,
    num_asks: int,
    seed: int | None = None,
) -> tuple[OrderbookLevels, OrderbookLevels]:
    """Generate a valid orderbook snapshot with random but valid levels.

    Args:
        num_bids: Number of bid levels to generate (1-50)
        num_asks: Number of ask levels to generate (1-50)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (asks, bids) OrderbookLevels
    """
    if seed is not None:
        random.seed(seed)

    # Generate bid prices: descending from base price
    base_bid = random.uniform(50.0, 100.0)
    # Ensure strictly descending order
    bid_prices = [base_bid]
    for i in range(1, num_bids):
        next_price = bid_prices[-1] - random.uniform(0.01, 0.1)
        bid_prices.append(next_price)

    # Generate ask prices: ascending from above best bid
    base_ask = bid_prices[0] + random.uniform(0.01, 1.0)
    # Ensure strictly ascending order
    ask_prices = [base_ask]
    for i in range(1, num_asks):
        next_price = ask_prices[-1] + random.uniform(0.01, 0.1)
        ask_prices.append(next_price)

    # Generate sizes
    bid_sizes = [random.uniform(0.001, 1000.0) for _ in range(num_bids)]
    ask_sizes = [random.uniform(0.001, 1000.0) for _ in range(num_asks)]

    # Generate norders
    bid_norders = [random.randint(1, 100) for _ in range(num_bids)]
    ask_norders = [random.randint(1, 100) for _ in range(num_asks)]

    asks = OrderbookLevels.from_list_with_ticks_and_lots(
        ask_prices, ask_sizes, ask_norders, TICK_SIZE, LOT_SIZE
    )
    bids = OrderbookLevels.from_list_with_ticks_and_lots(
        bid_prices, bid_sizes, bid_norders, TICK_SIZE, LOT_SIZE
    )

    return asks, bids


@pytest.mark.property
class TestBBOInvariant:
    """Test that best_bid <= best_ask invariant holds."""

    @pytest.mark.parametrize("seed", range(20))
    def test_bbo_invariant_after_snapshot(self, seed):
        """After any valid snapshot, best_bid <= best_ask."""
        book = _mk_book(num_levels=64)

        num_bids = random.randint(5, 30)
        num_asks = random.randint(5, 30)

        asks, bids = _generate_valid_snapshot(num_bids, num_asks, seed=seed)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        best_bid = bids_arr["price"][0]
        best_ask = asks_arr["price"][0]

        # Invariant: best_bid <= best_ask (or equal at zero spread)
        assert best_bid <= best_ask, (
            f"BBO invariant violated: bid={best_bid}, ask={best_ask}"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_bbo_invariant_after_deltas(self, seed):
        """After delta updates, BBO invariant maintained."""
        random.seed(seed)
        book = _mk_book(num_levels=64)

        # Initial snapshot
        asks, bids = _generate_valid_snapshot(10, 10, seed=seed)
        book.consume_snapshot(asks, bids)

        # Apply random deltas
        for _ in range(10):
            delta_price = random.uniform(99.0, 101.0)
            delta_size = random.uniform(0.1, 10.0)

            delta = OrderbookLevels.from_list_with_ticks_and_lots(
                [delta_price], [delta_size], [1], TICK_SIZE, LOT_SIZE
            )
            book.consume_deltas(delta, _empty_bid_levels())

        try:
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()

            best_bid = bids_arr["price"][0]
            best_ask = asks_arr["price"][0]

            assert best_bid <= best_ask
        except RuntimeError:
            # Empty orderbook after deltas is acceptable
            pass


@pytest.mark.property
class TestCapacityInvariant:
    """Test that capacity is never exceeded."""

    @pytest.mark.parametrize(
        "capacity,snapshot_size",
        [
            (64, 32),
            (64, 64),
            (64, 128),
            (128, 200),
            (256, 500),
        ],
    )
    def test_capacity_never_exceeded_snapshot(self, capacity, snapshot_size):
        """Orderbook never stores more than max_levels per side."""
        book = _mk_book(num_levels=capacity)

        # Generate large snapshot
        ask_prices = [100.0 + i * 0.01 for i in range(snapshot_size)]
        bid_prices = [99.99 - i * 0.01 for i in range(snapshot_size)]
        sizes = [1.0] * snapshot_size
        norders = [1] * snapshot_size

        asks = OrderbookLevels.from_list_with_ticks_and_lots(
            ask_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )
        bids = OrderbookLevels.from_list_with_ticks_and_lots(
            bid_prices, sizes, norders, TICK_SIZE, LOT_SIZE
        )

        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        # Invariant: never exceed capacity
        assert len(asks_arr) <= capacity
        assert len(bids_arr) <= capacity

    @pytest.mark.parametrize("seed", range(10))
    def test_capacity_never_exceeded_deltas(self, seed):
        """Capacity maintained through delta updates."""
        random.seed(seed)
        capacity = 64
        book = _mk_book(num_levels=capacity)

        # Initial snapshot at capacity
        asks, bids = _generate_valid_snapshot(capacity, capacity, seed=seed)
        book.consume_snapshot(asks, bids)

        # Random delta updates
        for i in range(50):
            price = random.uniform(90.0, 110.0)
            size = random.uniform(0.1, 10.0)

            delta = OrderbookLevels.from_list_with_ticks_and_lots(
                [price], [size], [1], TICK_SIZE, LOT_SIZE
            )
            book.consume_deltas(delta, _empty_bid_levels())

            # Check capacity after each delta
            try:
                asks_arr = book.get_asks_numpy()
                assert len(asks_arr) <= capacity, f"Capacity exceeded at iteration {i}"
            except RuntimeError:
                # Empty is valid
                pass


@pytest.mark.property
class TestSortedOrderInvariant:
    """Test that sorted order is maintained."""

    @pytest.mark.parametrize("seed", range(15))
    def test_bids_descending_after_snapshot(self, seed):
        """Bids always in descending price order."""
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(15, 15, seed=seed)
        book.consume_snapshot(asks, bids)

        bids_arr = book.get_bids_numpy()
        bid_prices = bids_arr["price"]

        # Check descending order
        for i in range(len(bid_prices) - 1):
            assert bid_prices[i] >= bid_prices[i + 1], (
                f"Bids not descending at index {i}"
            )

    @pytest.mark.parametrize("seed", range(15))
    def test_asks_ascending_after_snapshot(self, seed):
        """Asks always in ascending price order."""
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(15, 15, seed=seed)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        ask_prices = asks_arr["price"]

        # Check ascending order
        for i in range(len(ask_prices) - 1):
            assert ask_prices[i] <= ask_prices[i + 1], (
                f"Asks not ascending at index {i}"
            )

    @pytest.mark.parametrize("seed", range(10))
    def test_sorted_order_after_deltas(self, seed):
        """Sorted order maintained after delta updates."""
        random.seed(seed)
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(10, 10, seed=seed)
        book.consume_snapshot(asks, bids)

        # Apply random deltas
        for _ in range(20):
            price = random.uniform(95.0, 105.0)
            size = random.uniform(0.1, 10.0)

            delta = OrderbookLevels.from_list_with_ticks_and_lots(
                [price], [size], [1], TICK_SIZE, LOT_SIZE
            )
            book.consume_deltas(delta, _empty_bid_levels())

        try:
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()

            # Check sorted order
            ask_prices = asks_arr["price"]
            bid_prices = bids_arr["price"]

            for i in range(len(ask_prices) - 1):
                assert ask_prices[i] <= ask_prices[i + 1]

            for i in range(len(bid_prices) - 1):
                assert bid_prices[i] >= bid_prices[i + 1]
        except RuntimeError:
            # Empty is acceptable
            pass


@pytest.mark.property
class TestSnapshotClearEquivalence:
    """Test that snapshot → clear ≡ fresh orderbook."""

    def test_snapshot_then_clear_equivalent_to_new_book(self):
        """Snapshot → Clear ≡ Fresh orderbook."""
        book1 = _mk_book(num_levels=64)
        book2 = _mk_book(num_levels=64)

        # book1: apply snapshot then clear
        asks, bids = _generate_valid_snapshot(5, 5, seed=42)
        book1.consume_snapshot(asks, bids)
        book1.clear()

        # book2: fresh, never populated
        # Both should behave identically (return empty arrays)

        bids1 = book1.get_bids_numpy()
        bids2 = book2.get_bids_numpy()

        assert len(bids1) == 0
        assert len(bids2) == 0

        # get_mid_price should raise on both
        with pytest.raises(RuntimeError):
            book1.get_mid_price()

        with pytest.raises(RuntimeError):
            book2.get_mid_price()

    def test_multiple_clear_operations_idempotent(self):
        """Calling clear() multiple times is safe."""
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(5, 5, seed=123)
        book.consume_snapshot(asks, bids)

        # Clear multiple times
        book.clear()
        book.clear()
        book.clear()

        # Should still be empty
        bids_arr = book.get_bids_numpy()
        assert len(bids_arr) == 0


@pytest.mark.property
class TestOperationSequenceConsistency:
    """Test that sequences of operations maintain consistency."""

    @pytest.mark.parametrize("seed", range(5))
    def test_random_sequence_maintains_invariants(self, seed):
        """10-100 random ops maintain consistency."""
        random.seed(seed)
        book = _mk_book(num_levels=64)

        num_operations = random.randint(10, 100)

        for i in range(num_operations):
            operation = random.choice(["snapshot", "delta", "clear"])

            if operation == "snapshot":
                num_bids = random.randint(1, 15)
                num_asks = random.randint(1, 15)
                asks, bids = _generate_valid_snapshot(num_bids, num_asks, seed=seed + i)
                book.consume_snapshot(asks, bids)

            elif operation == "delta":
                price = random.uniform(90.0, 110.0)
                size = random.uniform(0.0, 10.0)  # Include deletions
                norders = 0 if size == 0.0 else 1

                delta = OrderbookLevels.from_list_with_ticks_and_lots(
                    [price], [size], [norders], TICK_SIZE, LOT_SIZE
                )
                book.consume_deltas(delta, _empty_bid_levels())

            elif operation == "clear":
                book.clear()

            # Check invariants after each operation
            try:
                asks_arr = book.get_asks_numpy()
                bids_arr = book.get_bids_numpy()

                # Capacity invariant
                assert len(asks_arr) <= 64
                assert len(bids_arr) <= 64

                # Sorted order invariant
                if len(asks_arr) > 1:
                    ask_prices = asks_arr["price"]
                    for j in range(len(ask_prices) - 1):
                        assert ask_prices[j] <= ask_prices[j + 1]

                if len(bids_arr) > 1:
                    bid_prices = bids_arr["price"]
                    for j in range(len(bid_prices) - 1):
                        assert bid_prices[j] >= bid_prices[j + 1]

                # BBO invariant
                if len(asks_arr) > 0 and len(bids_arr) > 0:
                    assert bids_arr["price"][0] <= asks_arr["price"][0]

            except RuntimeError:
                # Empty orderbook is valid
                pass

    def test_interleaved_snapshot_and_delta_updates(self):
        """Alternating snapshots and deltas."""
        book = _mk_book(num_levels=64)

        for i in range(20):
            if i % 2 == 0:
                # Snapshot
                asks, bids = _generate_valid_snapshot(8, 8, seed=i * 10)
                book.consume_snapshot(asks, bids)
            else:
                # Delta
                price = 100.0 + (i % 10) * 0.01
                size = 1.0 if i % 3 != 0 else 0.0  # Some deletions
                norders = 0 if size == 0.0 else 1

                delta = OrderbookLevels.from_list_with_ticks_and_lots(
                    [price], [size], [norders], TICK_SIZE, LOT_SIZE
                )
                book.consume_deltas(delta, _empty_bid_levels())

            # Verify consistency
            try:
                asks_arr = book.get_asks_numpy()
                bids_arr = book.get_bids_numpy()
                assert len(asks_arr) <= 64
                assert len(bids_arr) <= 64
            except RuntimeError:
                pass

    @pytest.mark.slow
    def test_stress_rapid_updates(self):
        """Stress test with 1000 rapid sequential updates."""
        book = _mk_book(num_levels=64)

        # Initial snapshot
        asks, bids = _generate_valid_snapshot(10, 10, seed=999)
        book.consume_snapshot(asks, bids)

        # Rapid updates
        for i in range(1000):
            price = 100.0 + (i % 50) * 0.01
            size = random.uniform(0.0, 5.0)
            norders = 0 if size == 0.0 else random.randint(1, 10)

            delta = OrderbookLevels.from_list_with_ticks_and_lots(
                [price], [size], [norders], TICK_SIZE, LOT_SIZE
            )
            book.consume_deltas(delta, _empty_bid_levels())

        # Should still be valid
        try:
            asks_arr = book.get_asks_numpy()
            assert len(asks_arr) <= 64
        except RuntimeError:
            # Empty is acceptable
            pass


@pytest.mark.property
class TestDataIntegrityInvariants:
    """Test that data integrity is maintained."""

    def test_no_negative_prices(self):
        """All prices must be positive."""
        book = _mk_book(num_levels=64)

        # Valid snapshot
        asks, bids = _generate_valid_snapshot(5, 5, seed=777)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        assert np.all(asks_arr["price"] > 0)
        assert np.all(bids_arr["price"] > 0)

    def test_no_negative_sizes(self):
        """All sizes must be non-negative."""
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(5, 5, seed=888)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        assert np.all(asks_arr["size"] >= 0)
        assert np.all(bids_arr["size"] >= 0)

    def test_no_zero_prices(self):
        """Orderbook should not contain levels with price=0."""
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(5, 5, seed=111)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        # No zero prices
        assert np.all(asks_arr["price"] != 0)
        assert np.all(bids_arr["price"] != 0)

    @pytest.mark.parametrize("seed", range(10))
    def test_norders_consistency(self, seed):
        """When size > 0, norders should be > 0."""
        book = _mk_book(num_levels=64)

        asks, bids = _generate_valid_snapshot(8, 8, seed=seed)
        book.consume_snapshot(asks, bids)

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()

        # All levels with size > 0 should have norders > 0
        for level in asks_arr:
            if level["size"] > 0:
                assert level["norders"] > 0

        for level in bids_arr:
            if level["size"] > 0:
                assert level["norders"] > 0

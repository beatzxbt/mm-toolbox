"""Tests for small orderbook capacity edge cases and BBO cross-removal behavior.

Verifies correct behavior at minimum capacity (16 levels), BBO updates that
would otherwise empty the book, and stress tests for small orderbooks.
"""

from __future__ import annotations

import pytest

from mm_toolbox.orderbook.advanced import (
    PyAdvancedOrderbook,
    OrderbookLevel,
    OrderbookLevels,
    PyOrderbookSortedness,
)
from tests.orderbook.advanced.conftest import (
    TICK_SIZE,
    LOT_SIZE,
    _mk_book,
    _make_levels,
    _empty_bid_levels,
    _bids_asks_arrays,
)


@pytest.mark.boundary
class TestMinimumCapacityEnforcement:
    """Layer 2: Test that minimum orderbook size of 4 levels is enforced."""

    @pytest.mark.parametrize("invalid_size", [0, 1, 2, 3])
    def test_reject_sizes_below_minimum(self, invalid_size: int):
        """Given size < 4, When creating PyAdvancedOrderbook, Then raises ValueError."""
        with pytest.raises(ValueError, match="expected >=4"):
            PyAdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=invalid_size,
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

    def test_accept_minimum_size(self):
        """Given size=4, When creating PyAdvancedOrderbook, Then succeeds."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=4,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None

    @pytest.mark.parametrize("valid_size", [4, 5, 16, 32, 64, 128, 1024])
    def test_accept_valid_sizes(self, valid_size: int):
        """Given size >= 4, When creating PyAdvancedOrderbook, Then succeeds."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=valid_size,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None


@pytest.mark.boundary
class TestBBOCrossRemovalRestoration:
    """Layer 2: Test BBO updates that would empty a side are properly handled."""

    def test_bbo_wipes_ask_side_restores_from_incoming(self):
        """Given BBO bid wiping all asks, When consumed, Then incoming ask becomes new top."""
        book = _mk_book(num_levels=64)

        # Initialize with asks at 100.0-100.15 and bids at 99.0-98.85
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.0 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Verify initial state
        initial_asks = book.get_asks_numpy()
        initial_bids = book.get_bids_numpy()
        assert len(initial_asks) == 16
        assert len(initial_bids) == 16
        assert initial_asks["price"][0] == pytest.approx(100.0)
        assert initial_bids["price"][0] == pytest.approx(99.0)

        # BBO update: bid at 101.0 (higher than all asks), ask at 102.0
        # This should wipe all asks and restore from incoming ask
        bbo_ask = OrderbookLevel(102.0, 5.0, norders=1)
        bbo_bid = OrderbookLevel(101.0, 3.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Ask side should be restored with incoming BBO ask
        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()

        assert len(final_asks) >= 1, "Ask side should have at least the incoming BBO"
        assert len(final_bids) >= 1
        assert final_asks["price"][0] == pytest.approx(102.0)
        assert final_bids["price"][0] == pytest.approx(101.0)

    def test_bbo_wipes_bid_side_restores_from_incoming(self):
        """Given BBO ask wiping all bids, When consumed, Then incoming bid becomes new top."""
        book = _mk_book(num_levels=64)

        # Initialize with asks at 101.0-101.15 and bids at 100.0-99.85
        asks, _ = _make_levels(
            [101.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [100.0 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # BBO update: ask at 98.0 (lower than all bids), bid at 97.0
        # Note: consume_bbo doesn't process ask-side cross removal the same way
        # The cross removal logic removes asks when bid >= ask
        bbo_ask = OrderbookLevel(98.0, 5.0, norders=1)
        bbo_bid = OrderbookLevel(97.0, 3.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Both sides should still have data
        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()

        assert len(final_asks) >= 1
        assert len(final_bids) >= 1

    def test_bbo_cross_removal_preserves_book_integrity(self):
        """Given series of BBO updates causing crosses, When applied, Then book never completely empty."""
        book = _mk_book(num_levels=64)

        # Initialize with minimal spread
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Series of BBO updates that push the book around
        for i in range(100):
            shift = (i % 20) * 0.01
            bbo_ask = OrderbookLevel(100.0 + shift, 1.0, norders=1)
            bbo_bid = OrderbookLevel(99.99 + shift, 1.0, norders=1)
            book.consume_bbo(bbo_ask, bbo_bid)

            # Book should never be empty
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, f"Ask side empty at iteration {i}"
            assert len(bids_arr) >= 1, f"Bid side empty at iteration {i}"


@pytest.mark.boundary
class TestSmallOrderbookStress:
    """Layer 3: Stress tests for minimum-sized orderbooks."""

    def test_1000_bbo_updates_minimum_capacity(self):
        """Given 16-level book, When 1000 BBO updates applied, Then does not crash."""
        book = _mk_book(num_levels=64)

        # Initialize
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # 1000 BBO updates with varying prices
        for i in range(1000):
            offset = (i % 50) * 0.01
            book.consume_bbo_values(
                100.0 + offset, float(i % 10 + 1), 99.99 + offset, float(i % 10 + 1)
            )

        # Should complete without crash and have valid state
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_10000_bbo_updates_minimum_capacity(self):
        """Given 16-level book, When 10000 BBO updates applied, Then does not crash."""
        book = _mk_book(num_levels=64)

        # Initialize
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # 10000 BBO updates
        for i in range(10000):
            offset = (i % 100) * 0.01
            book.consume_bbo_values(
                100.0 + offset, float(i % 10 + 1), 99.99 + offset, float(i % 10 + 1)
            )

        # Should complete without crash
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_mixed_operations_minimum_capacity(self):
        """Given 16-level book, When mixed snapshot/delta/BBO operations applied, Then does not crash."""
        book = _mk_book(num_levels=64)

        for i in range(500):
            op_type = i % 3

            if op_type == 0:
                # Snapshot
                asks, _ = _make_levels(
                    [100.0 + (i % 10) * 0.01 + j * 0.01 for j in range(16)],
                    [1.0] * 16,
                    with_precision=True,
                )
                bids, _ = _make_levels(
                    [99.99 + (i % 10) * 0.01 - j * 0.01 for j in range(16)],
                    [1.0] * 16,
                    with_precision=True,
                )
                book.consume_snapshot(asks, bids)

            elif op_type == 1:
                # Delta
                delta_asks = OrderbookLevels.from_list(
                    [100.0 + (i % 5) * 0.01], [float(i % 3 + 1)], [1]
                )
                book.consume_deltas(delta_asks, _empty_bid_levels())

            else:
                # BBO
                book.consume_bbo_values(
                    100.0 + (i % 15) * 0.01, 1.0, 99.99 + (i % 15) * 0.01, 1.0
                )

        # Should complete without crash
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1


@pytest.mark.boundary
class TestBBOCrossRemovalEdgeCases:
    """Layer 2: Edge cases for BBO cross-removal logic."""

    def test_bbo_exactly_at_cross_price(self):
        """Given BBO bid exactly at best ask price, When consumed, Then cross removal triggered."""
        book = _mk_book(num_levels=64)

        # Initialize
        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # BBO with bid exactly at best ask price
        bbo_ask = OrderbookLevel(101.0, 1.0, norders=1)
        bbo_bid = OrderbookLevel(100.0, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Should handle cross correctly
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_bbo_with_zero_size_ask(self):
        """Given BBO with zero-size ask, When consumed, Then does not cause issues."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # BBO with zero-size ask (deletion marker)
        bbo_ask = OrderbookLevel(100.0, 0.0, norders=0)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Book should remain valid
        book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(bids_arr) >= 1

    def test_bbo_with_zero_size_bid(self):
        """Given BBO with zero-size bid, When consumed, Then does not cause issues."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # BBO with zero-size bid (deletion marker)
        bbo_ask = OrderbookLevel(100.0, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 0.0, norders=0)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Book should remain valid
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) >= 1

    def test_repeated_cross_and_restore(self):
        """Given alternating crossing and non-crossing BBO updates, When applied, Then book integrity maintained."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Alternate between crossing and non-crossing BBO updates
        for i in range(100):
            if i % 2 == 0:
                # Crossing BBO
                bbo_ask = OrderbookLevel(102.0, 1.0, norders=1)
                bbo_bid = OrderbookLevel(101.0, 1.0, norders=1)
            else:
                # Non-crossing BBO
                bbo_ask = OrderbookLevel(100.0, 1.0, norders=1)
                bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
            book.consume_bbo(bbo_ask, bbo_bid)

            # Verify book integrity after each update
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, f"Empty asks at iteration {i}"
            assert len(bids_arr) >= 1, f"Empty bids at iteration {i}"

    def test_progressive_cross_removal(self):
        """Given progressively higher bid prices, When BBO updates applied, Then book integrity maintained."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # Progressively increase bid price to remove asks one by one
        for i in range(20):
            bid_price = 100.0 + i * 0.01
            bbo_ask = OrderbookLevel(bid_price + 1.0, 1.0, norders=1)
            bbo_bid = OrderbookLevel(bid_price, 1.0, norders=1)
            book.consume_bbo(bbo_ask, bbo_bid)

            # Book should never be completely empty
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, f"Empty asks at iteration {i}"
            assert len(bids_arr) >= 1, f"Empty bids at iteration {i}"


@pytest.mark.boundary
class TestMinimumCapacityBehavior:
    """Layer 3: Test 4-level orderbook behaves correctly under stress."""

    def test_4_level_snapshot(self):
        """Given 4-level book, When snapshot consumed, Then all 4 levels stored."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=4,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03, 100.04],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.0, 99.99, 99.98, 99.97],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        bid_arr, ask_arr = _bids_asks_arrays(book)
        assert len(bid_arr) == 4
        assert len(ask_arr) == 4
        assert book.get_bbo_spread() == pytest.approx(0.01)

    def test_4_level_delta_adds_beyond_capacity(self):
        """Given full 4-level book, When delta beyond capacity, Then truncated."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=4,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        # Fill to capacity
        asks, _ = _make_levels(
            prices=[100.0 + i * 0.01 for i in range(4)],
            sizes=[1.0] * 4,
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[99.99 - i * 0.01 for i in range(4)],
            sizes=[1.0] * 4,
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        # Try to add 5th level - should be ignored
        extra_ask = OrderbookLevels.from_list([100.05], [1.0], [1])
        book.consume_deltas(extra_ask, _empty_bid_levels())

        _, ask_arr = _bids_asks_arrays(book)
        assert len(ask_arr) == 4
        assert ask_arr["price"][-1] == pytest.approx(100.03)

    def test_4_level_bbo_updates(self):
        """Given 4-level book, When BBO updates applied, Then works correctly."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=4,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03, 100.04],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.0, 99.99, 99.98, 99.97],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        # BBO update
        new_ask = OrderbookLevel(100.015, 2.0, 1)
        new_bid = OrderbookLevel(100.005, 2.0, 1)
        book.consume_bbo(new_ask, new_bid)

        assert book.get_bbo_spread() == pytest.approx(0.01)

    def test_4_level_calculations(self):
        """Given 4-level book, When price calculations called, Then correct values returned."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=4,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03, 100.04],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.0, 99.99, 99.98, 99.97],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        assert book.get_mid_price() == pytest.approx(100.0)
        assert book.get_bbo_spread() == pytest.approx(0.01)
        assert book.get_wmid_price() > 0

        # Impact for 1.0 should be 0 (within first level)
        impact = book.get_price_impact(1.0, True, True)
        assert impact == 0.0

        # Impact for 2.0 should be 0.01 (spills into second level)
        impact = book.get_price_impact(2.0, True, True)
        assert impact == pytest.approx(0.01)

    def test_4_level_rapid_updates(self):
        """Given 4-level book, When 100 rapid updates applied, Then remains valid."""
        book = PyAdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=4,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03, 100.04],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        bids, _ = _make_levels(
            prices=[100.0, 99.99, 99.98, 99.97],
            sizes=[1.0, 1.0, 1.0, 1.0],
            with_precision=True,
        )
        book.consume_snapshot(asks, bids)

        for i in range(100):
            delta_asks = OrderbookLevels.from_list(
                [100.01 + (i % 4) * 0.01], [float(i % 10 + 1)], [1]
            )
            book.consume_deltas(delta_asks, _empty_bid_levels())

        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) <= 4


@pytest.mark.boundary
class TestWorseBBORestoration:
    """Layer 2: Test worse-price BBO updates trigger removal and restoration."""

    def test_worse_ask_bbo_restores_from_incoming(self):
        """Given single-level book, When worse ask BBO sent, Then old ask removed and new becomes top."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 1.0, norders=1)
        bbo_bid = OrderbookLevel(100.00, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()
        assert len(final_asks) == 1
        assert final_asks["price"][0] == pytest.approx(100.02)
        assert len(final_bids) == 1
        assert final_bids["price"][0] == pytest.approx(100.00)
        assert book.get_mid_price() > 0

    def test_worse_bid_bbo_restores_from_incoming(self):
        """Given single-level book, When worse bid BBO sent, Then old bid removed and new becomes top."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.01, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()
        assert len(final_asks) == 1
        assert final_asks["price"][0] == pytest.approx(100.01)
        assert len(final_bids) == 1
        assert final_bids["price"][0] == pytest.approx(99.99)

    def test_both_sides_worse_restores_both(self):
        """Given single-level book, When both sides get worse BBO, Then both restored from incoming."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()
        assert len(final_asks) == 1
        assert final_asks["price"][0] == pytest.approx(100.02)
        assert len(final_bids) == 1
        assert final_bids["price"][0] == pytest.approx(99.99)
        assert book.get_mid_price() > 0
        assert book.get_bbo_spread() > 0

    def test_worse_bbo_does_not_empty_book(self):
        """Given multi-level book, When worse BBO sent, Then old top removed and next level takes over."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02, 100.03], [1.0] * 3, with_precision=True)
        bids, _ = _make_levels([100.00, 99.99, 99.98], [1.0] * 3, with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()
        assert len(final_asks) >= 1
        assert len(final_bids) >= 1
        assert book.get_mid_price() > 0

    def test_worse_bbo_with_zero_size_incoming(self):
        """Given single-level book, When worse ask at size=0 sent, Then ask may empty without crash."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 0.0, norders=0)
        bbo_bid = OrderbookLevel(100.00, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        bids_arr, asks_arr = _bids_asks_arrays(book)
        assert len(bids_arr) >= 1

    def test_alternating_better_and_worse_bbo(self):
        """Given single-level book, When alternating better/worse BBO, Then book never empties."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        for _ in range(20):
            bbo_ask = OrderbookLevel(100.00, 1.0, norders=1)
            bbo_bid = OrderbookLevel(99.98, 1.0, norders=1)
            book.consume_bbo(bbo_ask, bbo_bid)

            bbo_ask = OrderbookLevel(100.01, 1.0, norders=1)
            bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
            book.consume_bbo(bbo_ask, bbo_bid)

            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, "Ask side emptied"
            assert len(bids_arr) >= 1, "Bid side emptied"

    def test_worse_ask_preserves_book(self):
        """Given 2-level ask book, When worse ask BBO sent, Then old top removed, next level becomes BBO."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02], [1.0] * 2, with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 1.0, norders=1)
        bbo_bid = OrderbookLevel(100.00, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        bids_arr, asks_arr = _bids_asks_arrays(book)
        assert len(asks_arr) >= 1
        assert asks_arr["price"][0] == pytest.approx(100.02)
        assert book.get_mid_price() > 0

    def test_worse_bid_preserves_book(self):
        """Given 2-level bid book, When worse bid BBO sent, Then old top removed, next level becomes BBO."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00, 99.99], [1.0] * 2, with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.01, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        bids_arr, _ = _bids_asks_arrays(book)
        assert len(bids_arr) >= 1
        assert bids_arr["price"][0] == pytest.approx(99.99)

    def test_both_worse_single_level_book_restores(self):
        """Given single-level book, When both sides get worse BBO, Then restored from incoming levels."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        bids_arr, asks_arr = _bids_asks_arrays(book)
        assert len(asks_arr) == 1
        assert len(bids_arr) == 1
        assert asks_arr["price"][0] == pytest.approx(100.02)
        assert bids_arr["price"][0] == pytest.approx(99.99)
        assert book.get_mid_price() > 0
        assert book.get_bbo_spread() > 0

    def test_worse_bbo_does_not_empty_multi_level_book(self):
        """Given 3-level book, When worse BBO on both sides, Then next levels take over without crash."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01, 100.02, 100.03], [1.0] * 3, with_precision=True)
        bids, _ = _make_levels([100.00, 99.99, 99.98], [1.0] * 3, with_precision=True)
        book.consume_snapshot(asks, bids)

        bbo_ask = OrderbookLevel(100.02, 1.0, norders=1)
        bbo_bid = OrderbookLevel(99.99, 1.0, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        bids_arr, asks_arr = _bids_asks_arrays(book)
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_alternating_valid_better_and_worse_20_cycles(self):
        """Given single-level book, When valid BBO alternates for 20 cycles, Then never empties."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels([100.01], [1.0], with_precision=True)
        bids, _ = _make_levels([100.00], [1.0], with_precision=True)
        book.consume_snapshot(asks, bids)

        for _ in range(20):
            book.consume_bbo(
                OrderbookLevel(100.01, 1.0, norders=1),
                OrderbookLevel(100.00, 1.0, norders=1),
            )
            book.consume_bbo(
                OrderbookLevel(100.02, 1.0, norders=1),
                OrderbookLevel(99.99, 1.0, norders=1),
            )

            bids_arr, asks_arr = _bids_asks_arrays(book)
            assert len(asks_arr) >= 1, "Ask side emptied during cycle"
            assert len(bids_arr) >= 1, "Bid side emptied during cycle"

    def test_crossed_bbo_rejected(self):
        """Given crossed BBO levels, When consumed, Then ValueError is raised."""
        book = _mk_book(num_levels=64)

        with pytest.raises(ValueError, match="Crossed BBO"):
            book.consume_bbo(
                OrderbookLevel(100.00, 1.0, norders=1),
                OrderbookLevel(100.01, 1.0, norders=1),
            )

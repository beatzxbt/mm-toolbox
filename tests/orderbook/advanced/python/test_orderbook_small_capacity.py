"""
Tests for small orderbook capacity edge cases and BBO cross-removal behavior.

Verifies correct behavior at minimum capacity (16 levels), BBO updates that
would otherwise empty the book, and stress tests for small orderbooks.
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
    _empty_bid_levels,
)


@pytest.mark.boundary
class TestMinimumCapacityEnforcement:
    """Test that minimum orderbook size of 16 levels is enforced."""

    @pytest.mark.parametrize("invalid_size", [0, 1, 2, 4, 8, 16, 32, 63])
    def test_reject_sizes_below_minimum(self, invalid_size: int):
        """Orderbook creation fails for sizes below 16."""
        with pytest.raises(ValueError, match="expected >=64"):
            AdvancedOrderbook(
                tick_size=TICK_SIZE,
                lot_size=LOT_SIZE,
                num_levels=invalid_size,
                delta_sortedness=PyOrderbookSortedness.UNKNOWN,
                snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
            )

    def test_accept_minimum_size(self):
        """Orderbook creation succeeds at minimum size of 16."""
        book = AdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=64,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None

    @pytest.mark.parametrize("valid_size", [64, 65, 128, 256, 512, 1024])
    def test_accept_valid_sizes(self, valid_size: int):
        """Orderbook creation succeeds for sizes >= 16."""
        book = AdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=valid_size,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None


@pytest.mark.boundary
class TestBBOCrossRemovalRestoration:
    """Test BBO updates that would empty a side are properly handled."""

    def test_bbo_wipes_ask_side_restores_from_incoming(self):
        """When BBO bid wipes all asks, incoming ask becomes new top."""
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(102.0, 5.0, TICK_SIZE, LOT_SIZE, norders=1)
        bbo_bid = OrderbookLevel.with_ticks_and_lots(101.0, 3.0, TICK_SIZE, LOT_SIZE, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Ask side should be restored with incoming BBO ask
        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()

        assert len(final_asks) >= 1, "Ask side should have at least the incoming BBO"
        assert len(final_bids) >= 1
        assert final_asks["price"][0] == pytest.approx(102.0)
        assert final_bids["price"][0] == pytest.approx(101.0)

    def test_bbo_wipes_bid_side_restores_from_incoming(self):
        """When BBO ask wipes all bids, incoming bid becomes new top."""
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(98.0, 5.0, TICK_SIZE, LOT_SIZE, norders=1)
        bbo_bid = OrderbookLevel.with_ticks_and_lots(97.0, 3.0, TICK_SIZE, LOT_SIZE, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Both sides should still have data
        final_asks = book.get_asks_numpy()
        final_bids = book.get_bids_numpy()

        assert len(final_asks) >= 1
        assert len(final_bids) >= 1

    def test_bbo_cross_removal_preserves_book_integrity(self):
        """Cross removal via BBO never leaves book completely empty."""
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
            bbo_ask = OrderbookLevel.with_ticks_and_lots(
                100.0 + shift, 1.0, TICK_SIZE, LOT_SIZE, norders=1
            )
            bbo_bid = OrderbookLevel.with_ticks_and_lots(
                99.99 + shift, 1.0, TICK_SIZE, LOT_SIZE, norders=1
            )
            book.consume_bbo(bbo_ask, bbo_bid)

            # Book should never be empty
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, f"Ask side empty at iteration {i}"
            assert len(bids_arr) >= 1, f"Bid side empty at iteration {i}"


@pytest.mark.boundary
class TestSmallOrderbookStress:
    """Stress tests for minimum-sized orderbooks."""

    def test_1000_bbo_updates_minimum_capacity(self):
        """1000 BBO updates on 16-level orderbook should not crash."""
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
            bbo_ask = OrderbookLevel.with_ticks_and_lots(
                100.0 + offset, float(i % 10 + 1), TICK_SIZE, LOT_SIZE, norders=1
            )
            bbo_bid = OrderbookLevel.with_ticks_and_lots(
                99.99 + offset, float(i % 10 + 1), TICK_SIZE, LOT_SIZE, norders=1
            )
            book.consume_bbo(bbo_ask, bbo_bid)

        # Should complete without crash and have valid state
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_10000_bbo_updates_minimum_capacity(self):
        """10000 BBO updates on 16-level orderbook should not crash."""
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
            bbo_ask = OrderbookLevel.with_ticks_and_lots(
                100.0 + offset, float(i % 10 + 1), TICK_SIZE, LOT_SIZE, norders=1
            )
            bbo_bid = OrderbookLevel.with_ticks_and_lots(
                99.99 + offset, float(i % 10 + 1), TICK_SIZE, LOT_SIZE, norders=1
            )
            book.consume_bbo(bbo_ask, bbo_bid)

        # Should complete without crash
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_mixed_operations_minimum_capacity(self):
        """Mix of snapshot, delta, and BBO operations on minimum capacity."""
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
                delta_asks = OrderbookLevels.from_list_with_ticks_and_lots(
                    [100.0 + (i % 5) * 0.01],
                    [float(i % 3 + 1)],
                    [1],
                    TICK_SIZE,
                    LOT_SIZE,
                )
                book.consume_deltas(delta_asks, _empty_bid_levels())

            else:
                # BBO
                bbo_ask = OrderbookLevel.with_ticks_and_lots(
                    100.0 + (i % 15) * 0.01, 1.0, TICK_SIZE, LOT_SIZE, norders=1
                )
                bbo_bid = OrderbookLevel.with_ticks_and_lots(
                    99.99 + (i % 15) * 0.01, 1.0, TICK_SIZE, LOT_SIZE, norders=1
                )
                book.consume_bbo(bbo_ask, bbo_bid)

        # Should complete without crash
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1


@pytest.mark.boundary
class TestBBOCrossRemovalEdgeCases:
    """Edge cases for BBO cross-removal logic."""

    def test_bbo_exactly_at_cross_price(self):
        """BBO bid equals best ask price triggers cross removal."""
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(101.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1)
        bbo_bid = OrderbookLevel.with_ticks_and_lots(100.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Should handle cross correctly
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) >= 1
        assert len(bids_arr) >= 1

    def test_bbo_with_zero_size_ask(self):
        """BBO with zero-size ask should not cause issues."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # BBO with zero-size ask (deletion marker)
        bbo_ask = OrderbookLevel.with_ticks_and_lots(100.0, 0.0, TICK_SIZE, LOT_SIZE, norders=0)
        bbo_bid = OrderbookLevel.with_ticks_and_lots(99.99, 1.0, TICK_SIZE, LOT_SIZE, norders=1)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Book should remain valid
        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(bids_arr) >= 1

    def test_bbo_with_zero_size_bid(self):
        """BBO with zero-size bid should not cause issues."""
        book = _mk_book(num_levels=64)

        asks, _ = _make_levels(
            [100.0 + i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        bids, _ = _make_levels(
            [99.99 - i * 0.01 for i in range(16)], [1.0] * 16, with_precision=True
        )
        book.consume_snapshot(asks, bids)

        # BBO with zero-size bid (deletion marker)
        bbo_ask = OrderbookLevel.with_ticks_and_lots(100.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1)
        bbo_bid = OrderbookLevel.with_ticks_and_lots(99.99, 0.0, TICK_SIZE, LOT_SIZE, norders=0)
        book.consume_bbo(bbo_ask, bbo_bid)

        # Book should remain valid
        asks_arr = book.get_asks_numpy()
        assert len(asks_arr) >= 1

    def test_repeated_cross_and_restore(self):
        """Repeated BBO updates that cross and restore."""
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
                bbo_ask = OrderbookLevel.with_ticks_and_lots(
                    102.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
                )
                bbo_bid = OrderbookLevel.with_ticks_and_lots(
                    101.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
                )
            else:
                # Non-crossing BBO
                bbo_ask = OrderbookLevel.with_ticks_and_lots(
                    100.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
                )
                bbo_bid = OrderbookLevel.with_ticks_and_lots(
                    99.99, 1.0, TICK_SIZE, LOT_SIZE, norders=1
                )
            book.consume_bbo(bbo_ask, bbo_bid)

            # Verify book integrity after each update
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, f"Empty asks at iteration {i}"
            assert len(bids_arr) >= 1, f"Empty bids at iteration {i}"

    def test_progressive_cross_removal(self):
        """BBO updates that progressively remove levels."""
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
            bbo_ask = OrderbookLevel.with_ticks_and_lots(
                bid_price + 1.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
            )
            bbo_bid = OrderbookLevel.with_ticks_and_lots(
                bid_price, 1.0, TICK_SIZE, LOT_SIZE, norders=1
            )
            book.consume_bbo(bbo_ask, bbo_bid)

            # Book should never be completely empty
            asks_arr = book.get_asks_numpy()
            bids_arr = book.get_bids_numpy()
            assert len(asks_arr) >= 1, f"Empty asks at iteration {i}"
            assert len(bids_arr) >= 1, f"Empty bids at iteration {i}"

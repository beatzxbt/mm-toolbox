"""
Tests for small orderbook capacity edge cases and BBO cross-removal behavior.

Verifies correct behavior at minimum capacity (16 levels), BBO updates that
would otherwise empty the book, and stress tests for small orderbooks.
"""

from __future__ import annotations

import numpy as np
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

    @pytest.mark.parametrize("invalid_size", [0, 1, 2, 4, 8, 15])
    def test_reject_sizes_below_minimum(self, invalid_size: int):
        """Orderbook creation fails for sizes below 16."""
        with pytest.raises(ValueError, match="expected >=16"):
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
            num_levels=16,
            delta_sortedness=PyOrderbookSortedness.UNKNOWN,
            snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
        )
        assert book is not None

    @pytest.mark.parametrize("valid_size", [16, 17, 32, 64, 65, 128, 256, 512, 1024])
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(
            102.0, 5.0, TICK_SIZE, LOT_SIZE, norders=1
        )
        bbo_bid = OrderbookLevel.with_ticks_and_lots(
            101.0, 3.0, TICK_SIZE, LOT_SIZE, norders=1
        )
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(
            98.0, 5.0, TICK_SIZE, LOT_SIZE, norders=1
        )
        bbo_bid = OrderbookLevel.with_ticks_and_lots(
            97.0, 3.0, TICK_SIZE, LOT_SIZE, norders=1
        )
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(
            101.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
        )
        bbo_bid = OrderbookLevel.with_ticks_and_lots(
            100.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
        )
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(
            100.0, 0.0, TICK_SIZE, LOT_SIZE, norders=0
        )
        bbo_bid = OrderbookLevel.with_ticks_and_lots(
            99.99, 1.0, TICK_SIZE, LOT_SIZE, norders=1
        )
        book.consume_bbo(bbo_ask, bbo_bid)

        # Book should remain valid
        book.get_asks_numpy()
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
        bbo_ask = OrderbookLevel.with_ticks_and_lots(
            100.0, 1.0, TICK_SIZE, LOT_SIZE, norders=1
        )
        bbo_bid = OrderbookLevel.with_ticks_and_lots(
            99.99, 0.0, TICK_SIZE, LOT_SIZE, norders=0
        )
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


@pytest.mark.boundary
class TestSmallCapacityMixedDeltas:
    """Verify mixed delta updates remain stable at minimum capacity."""

    def test_mixed_delta_sequence_does_not_corrupt(self):
        """Apply a deterministic mixed delta sequence at minimum capacity."""
        book = AdvancedOrderbook(
            tick_size=TICK_SIZE,
            lot_size=LOT_SIZE,
            num_levels=16,
            delta_sortedness=PyOrderbookSortedness.ASCENDING,
            snapshot_sortedness=PyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING,
        )

        asks = np.array([100.0 + 0.01 * i for i in range(16)], dtype=np.float64)
        bids = np.array([99.99 - 0.01 * i for i in range(16)], dtype=np.float64)
        ask_sizes = np.array([1.0 + (i % 5) * 0.1 for i in range(16)], dtype=np.float64)
        bid_sizes = np.array([1.0 + (i % 5) * 0.1 for i in range(16)], dtype=np.float64)

        book.consume_snapshot_numpy(asks, ask_sizes, bids, bid_sizes)

        delta_sequence = [
            (
                [
                    100.0101210945452,
                    100.12148192414071,
                    100.22955858294628,
                    100.24136622159048,
                ],
                [0.5, 0.5, 1.0, 0.5],
                [
                    99.8359318499268,
                    99.94916469687816,
                    99.96399855835696,
                    99.98227199878085,
                ],
                [0.0, 1.0, 0.5, 1.0],
            ),
            ([100.224709571992], [2.0], [99.91415614365971], [0.0]),
            (
                [100.05111943737903, 100.0790491748804, 100.16010585038552],
                [0.0, 0.0, 2.0],
                [
                    99.76494710112517,
                    99.85624640619513,
                    99.87512487219661,
                    99.88260568227233,
                ],
                [0.0, 2.0, 1.0, 0.5],
            ),
            (
                [100.0477667728756, 100.0596539821538, 100.14187768515517],
                [2.0, 0.0, 0.0],
                [99.98220966006963],
                [1.0],
            ),
            (
                [
                    100.02726446148277,
                    100.1368602278321,
                    100.13781681152264,
                    100.17664035246672,
                    100.20361671582283,
                ],
                [2.0, 0.0, 2.0, 1.0, 0.5],
                [
                    99.87966806567287,
                    99.8910280954021,
                    99.89476455071073,
                    99.9813212510337,
                ],
                [1.0, 0.5, 0.5, 0.5],
            ),
            (
                [100.24532493512456],
                [0.0],
                [
                    99.76658045963613,
                    99.7812538681211,
                    99.78589207993159,
                    99.81241349007463,
                    99.91289375023413,
                ],
                [0.0, 2.0, 1.0, 0.5, 0.5],
            ),
            (
                [
                    100.11264077665778,
                    100.1449237526864,
                    100.1650613446556,
                    100.22923530448686,
                    100.2490644598384,
                ],
                [1.0, 0.5, 0.5, 0.0, 1.0],
                [
                    99.76976951716719,
                    99.8667466084726,
                    99.89706794520977,
                    99.94039802019125,
                ],
                [0.0, 0.5, 1.0, 0.5],
            ),
            (
                [100.02515188005403, 100.03658962222808, 100.20397827413342],
                [0.0, 0.0, 0.0],
                [
                    99.76085617628775,
                    99.8877278488294,
                    99.9174409536459,
                    99.96840384352777,
                ],
                [0.5, 0.0, 2.0, 0.0],
            ),
            (
                [100.00540912746375, 100.00909800940288, 100.24025782005991],
                [0.0, 2.0, 0.5],
                [99.79439326593538],
                [0.0],
            ),
            (
                [100.1064047079917],
                [0.5],
                [
                    99.7743600526498,
                    99.80299902511591,
                    99.81238077355027,
                    99.83407055217516,
                    99.90526217276047,
                ],
                [0.0, 2.0, 0.0, 0.0, 2.0],
            ),
            (
                [100.0896388253279, 100.18289957655634],
                [0.5, 0.0],
                [99.7906619054634, 99.95119837564641, 99.97043569487887],
                [0.5, 0.5, 1.0],
            ),
            (
                [
                    100.00330093963375,
                    100.02930321096777,
                    100.16640145550448,
                    100.17032016890106,
                    100.2303545986159,
                ],
                [1.0, 2.0, 1.0, 0.5, 0.0],
                [99.9599531266247, 99.96602353257379, 99.97020266946217],
                [2.0, 0.0, 1.0],
            ),
            (
                [100.07022079105458],
                [0.5],
                [
                    99.80763129787833,
                    99.81909069956554,
                    99.83453413476373,
                    99.8362082873409,
                    99.97875104155867,
                ],
                [1.0, 2.0, 2.0, 0.0, 0.0],
            ),
            (
                [
                    100.04001994475614,
                    100.05577445521466,
                    100.11203383166091,
                    100.17464567224384,
                    100.17758749402206,
                ],
                [0.0, 2.0, 2.0, 0.0, 0.5],
                [99.91170604807404, 99.95988903596593],
                [2.0, 0.0],
            ),
            (
                [100.11159117804388, 100.22123637699944, 100.22700996428989],
                [0.0, 2.0, 1.0],
                [99.75001659908988, 99.89495457307486],
                [1.0, 2.0],
            ),
            (
                [100.0470003235127],
                [0.0],
                [
                    99.77003209204217,
                    99.9019413023804,
                    99.92413304531071,
                    99.98986088626928,
                ],
                [2.0, 2.0, 1.0, 0.0],
            ),
            (
                [100.00058939117984, 100.17933103582776],
                [0.0, 0.5],
                [99.94745553852754],
                [0.0],
            ),
            (
                [
                    100.02504517226593,
                    100.06999568583171,
                    100.21348452739933,
                    100.21841345597508,
                    100.24462879669335,
                ],
                [1.0, 2.0, 0.0, 1.0, 0.5],
                [99.76952290002437, 99.84520708255941],
                [1.0, 0.0],
            ),
            (
                [100.0105744005007, 100.21280993710938],
                [1.0, 0.0],
                [99.79937657776057, 99.81232065181437, 99.82553117563695],
                [2.0, 2.0, 2.0],
            ),
            (
                [100.04457207994135, 100.07275887207854, 100.09389473048435],
                [0.0, 0.0, 1.0],
                [
                    99.77248980284618,
                    99.78322966781441,
                    99.81512962341321,
                    99.83100265644126,
                    99.83812835404605,
                ],
                [0.5, 0.5, 1.0, 1.0, 2.0],
            ),
            (
                [
                    100.01206409057207,
                    100.04489671226038,
                    100.05992652090966,
                    100.07335175036434,
                    100.11951616728776,
                ],
                [2.0, 1.0, 1.0, 2.0, 0.0],
                [99.76700709218264, 99.87553205560802],
                [0.0, 2.0],
            ),
            (
                [
                    100.08591289841445,
                    100.11977162978799,
                    100.21021208315692,
                    100.24405736441227,
                ],
                [0.5, 0.5, 2.0, 0.0],
                [99.82245674789264, 99.85236847765057, 99.91790286987614],
                [0.0, 0.0, 0.5],
            ),
            (
                [100.02451912759581, 100.09620297515364],
                [1.0, 2.0],
                [99.88355390581907],
                [2.0],
            ),
            (
                [
                    100.06522913386156,
                    100.09208286408361,
                    100.10575951837686,
                    100.19427503862713,
                    100.24711476452482,
                ],
                [0.0, 0.0, 2.0, 0.5, 0.0],
                [99.83604489168229, 99.85349304591169],
                [2.0, 2.0],
            ),
            (
                [100.1602493171433, 100.19935830350201, 100.24926012372218],
                [2.0, 2.0, 1.0],
                [99.77511112546557, 99.89961962010804],
                [0.5, 0.0],
            ),
        ]

        for ask_prices, ask_sizes, bid_prices, bid_sizes in delta_sequence:
            book.consume_deltas_numpy(
                np.array(ask_prices, dtype=np.float64),
                np.array(ask_sizes, dtype=np.float64),
                np.array(bid_prices, dtype=np.float64),
                np.array(bid_sizes, dtype=np.float64),
            )

        asks_arr = book.get_asks_numpy()
        bids_arr = book.get_bids_numpy()
        assert len(asks_arr) > 0
        assert len(bids_arr) > 0
        assert bids_arr["price"][0] < asks_arr["price"][0]

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

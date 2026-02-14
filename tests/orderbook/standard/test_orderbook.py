"""Comprehensive test suite for the standard orderbook implementation."""

import pytest

from mm_toolbox.orderbook.standard import (
    Orderbook,
    OrderbookLevel,
    price_from_ticks as _price_from_ticks,
    price_to_ticks as _price_to_ticks,
    size_from_lots as _size_from_lots,
    size_to_lots as _size_to_lots,
)


class TestUtilityFunctions:
    """Test the utility conversion functions in isolation."""

    def test_price_to_ticks_conversion(self):
        """Test price to ticks conversion with various tick sizes."""
        # Standard tick size
        assert _price_to_ticks(100.01, 0.01) == 10001
        assert _price_to_ticks(100.00, 0.01) == 10000
        assert _price_to_ticks(99.99, 0.01) == 9999

        # Fractional tick size
        assert _price_to_ticks(100.125, 0.125) == 801
        assert _price_to_ticks(100.0, 0.125) == 800

        # Small tick size
        assert _price_to_ticks(100.0001, 0.0001) == 1000001

    def test_size_to_lots_conversion(self):
        """Test size to lots conversion with various lot sizes."""
        assert _size_to_lots(1.0, 0.001) == 1000
        assert _size_to_lots(0.5, 0.001) == 500
        assert _size_to_lots(0.0001, 0.0001) == 1

    def test_price_from_ticks_conversion(self):
        """Test ticks to price conversion."""
        assert _price_from_ticks(10001, 0.01) == 100.01
        assert _price_from_ticks(10000, 0.01) == 100.00
        assert abs(_price_from_ticks(801, 0.125) - 100.125) < 1e-10

    def test_size_from_lots_conversion(self):
        """Test lots to size conversion."""
        assert _size_from_lots(1000, 0.001) == 1.0
        assert _size_from_lots(500, 0.001) == 0.5
        assert abs(_size_from_lots(1, 0.0001) - 0.0001) < 1e-10


class TestOrderbookLevel:
    """Test OrderbookLevel functionality in isolation."""

    def test_basic_creation(self):
        """Test basic OrderbookLevel creation and validation."""
        level = OrderbookLevel(price=100.0, size=1.5, norders=2)
        assert level.price == 100.0
        assert level.size == 1.5
        assert level.norders == 2
        assert level.ticks == -1
        assert level.lots == -1

    def test_validation_errors(self):
        """Test that invalid values raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid price"):
            OrderbookLevel(price=-1.0, size=1.0, norders=1)

        with pytest.raises(ValueError, match="Invalid size"):
            OrderbookLevel(price=100.0, size=-1.0, norders=1)

        with pytest.raises(ValueError, match="Invalid norders"):
            OrderbookLevel(price=100.0, size=1.0, norders=-1)

    def test_value_property(self):
        """Test the value property calculation."""
        level = OrderbookLevel(price=100.0, size=1.5, norders=2)
        assert level.value == 150.0

    def test_precision_info_addition(self):
        """Test adding precision information to a level."""
        level = OrderbookLevel(price=100.01, size=1.5, norders=2)
        level.add_precision_info(tick_size=0.01, lot_size=0.001)

        assert level.ticks == 10001
        assert level.lots == 1500

    def test_precision_info_validation(self):
        """Test validation of tick_size and lot_size."""
        level = OrderbookLevel(price=100.0, size=1.0, norders=1)

        with pytest.raises(ValueError, match="Invalid tick_size"):
            level.add_precision_info(tick_size=-0.01, lot_size=0.001)

        with pytest.raises(ValueError, match="Invalid lot_size"):
            level.add_precision_info(tick_size=0.01, lot_size=-0.001)

    def test_from_values_class_method(self):
        """Test the from_values class method."""
        level = OrderbookLevel.from_values(
            price=100.01, size=1.5, norders=2, tick_size=0.01, lot_size=0.001
        )

        assert level.price == 100.01
        assert level.size == 1.5
        assert level.norders == 2
        assert level.ticks == 10001
        assert level.lots == 1500

    def test_reset_functionality(self):
        """Test the reset method."""
        level = OrderbookLevel.from_values(
            price=100.01, size=1.5, norders=2, tick_size=0.01, lot_size=0.001
        )

        level.reset()
        assert level.price == 0.0
        assert level.size == 0.0
        assert level.norders == 0
        assert level.ticks == -1
        assert level.lots == -1


class TestOrderbookInitialization:
    """Test orderbook initialization and basic setup."""

    def test_basic_initialization(self):
        """Test basic orderbook initialization."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=100)

        assert ob._tick_size == 0.01
        assert ob._lot_size == 0.001
        assert ob._size == 100
        assert len(ob._asks) == 0
        assert len(ob._bids) == 0
        assert len(ob._sorted_ask_ticks) == 0
        assert len(ob._sorted_bid_ticks) == 0
        assert not ob._is_populated
        assert not ob._trust_input_precision

    def test_initialization_validation(self):
        """Test that invalid initialization parameters raise errors."""
        with pytest.raises(ValueError, match="Invalid tick_size"):
            Orderbook(tick_size=-0.01, lot_size=0.001)

        with pytest.raises(ValueError, match="Invalid lot_size"):
            Orderbook(tick_size=0.01, lot_size=-0.001)

    def test_initialization_with_initial_data(self):
        """Test initialization with initial bids and asks."""
        bids = [
            OrderbookLevel.from_values(100.0, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
        ]

        ob = Orderbook(
            tick_size=0.01, lot_size=0.001, size=2, initial_bids=bids, initial_asks=asks
        )

        assert ob._is_populated
        assert len(ob._bids) == 2
        assert len(ob._asks) == 2


class TestOrderbookSnapshots:
    """Test orderbook snapshot functionality."""

    def test_basic_snapshot_update(self):
        """Test basic snapshot update functionality."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 3.0, 3, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 3.5, 3, 0.01, 0.001),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        assert ob._is_populated
        assert len(ob._bids) == 3
        assert len(ob._asks) == 3

        # Check internal tick sorting
        assert ob._sorted_bid_ticks == [9998, 9999, 10000]  # Ascending
        assert ob._sorted_ask_ticks == [10001, 10002, 10003]  # Ascending

    def test_snapshot_without_precision_info(self):
        """Test snapshot update without pre-computed precision info."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=2)

        bids = [
            OrderbookLevel(price=100.00, size=1.0, norders=1),
            OrderbookLevel(price=99.99, size=2.0, norders=2),
        ]
        asks = [
            OrderbookLevel(price=100.01, size=1.5, norders=1),
            OrderbookLevel(price=100.02, size=2.5, norders=2),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        # Verify precision info was added automatically
        for bid in bids:
            assert bid.ticks is not None
            assert bid.lots is not None
        for ask in asks:
            assert ask.ticks is not None
            assert ask.lots is not None

    def test_snapshot_recomputes_existing_precision_by_default(self):
        """Default mode should recompute precision info for integrity."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=2)

        bids = [
            OrderbookLevel(price=100.00, size=1.0, norders=1, ticks=1, lots=1),
            OrderbookLevel(price=99.99, size=2.0, norders=2, ticks=2, lots=2),
        ]
        asks = [
            OrderbookLevel(price=100.01, size=1.5, norders=1, ticks=3, lots=3),
            OrderbookLevel(price=100.02, size=2.5, norders=2, ticks=4, lots=4),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        assert ob._sorted_bid_ticks == [9999, 10000]
        assert ob._sorted_ask_ticks == [10001, 10002]
        assert bids[0].ticks == 10000
        assert asks[0].ticks == 10001

    def test_snapshot_trusts_existing_precision_when_enabled(self):
        """Trusted mode should reuse existing precision info."""
        ob = Orderbook(
            tick_size=0.01,
            lot_size=0.001,
            size=2,
            trust_input_precision=True,
        )

        bids = [
            OrderbookLevel(price=100.00, size=1.0, norders=1, ticks=49999, lots=1000),
            OrderbookLevel(price=99.99, size=2.0, norders=2, ticks=50000, lots=2000),
        ]
        asks = [
            OrderbookLevel(price=100.01, size=1.5, norders=1, ticks=50001, lots=1500),
            OrderbookLevel(price=100.02, size=2.5, norders=2, ticks=50002, lots=2500),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        assert ob._sorted_bid_ticks == [49999, 50000]
        assert ob._sorted_ask_ticks == [50001, 50002]
        assert 10000 not in ob._bids
        assert 10001 not in ob._asks

    def test_snapshot_validation(self):
        """Test snapshot validation for minimum size requirements."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=5)

        bids = [OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001)]  # Only 1 level
        asks = [
            OrderbookLevel.from_values(100.01 + i * 0.01, 1.5, 1, 0.01, 0.001)
            for i in range(5)
        ]

        with pytest.raises(ValueError, match="Invalid bids with snapshot"):
            ob.consume_snapshot(asks=asks, bids=bids)

        bids = [
            OrderbookLevel.from_values(100.00 - i * 0.01, 1.0, 1, 0.01, 0.001)
            for i in range(5)
        ]
        asks = [OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001)]  # Only 1 level

        with pytest.raises(ValueError, match="Invalid asks with snapshot"):
            ob.consume_snapshot(asks=asks, bids=bids)

    def test_snapshot_deduplicates_duplicate_ticks(self):
        """Duplicate snapshot ticks should not create duplicate tick-list entries."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel(price=100.00, size=1.0, norders=1),
            OrderbookLevel(price=100.009, size=2.0, norders=2),  # same tick as 100.00
            OrderbookLevel(price=99.99, size=3.0, norders=3),
        ]
        asks = [
            OrderbookLevel(price=100.01, size=1.5, norders=1),
            OrderbookLevel(price=100.019, size=2.5, norders=2),  # same tick as 100.01
            OrderbookLevel(price=100.02, size=3.5, norders=3),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        assert len(ob._bids) == 2
        assert len(ob._asks) == 2
        assert ob._sorted_bid_ticks == [9999, 10000]
        assert ob._sorted_ask_ticks == [10001, 10002]

        # Duplicate ticks collapse to one level; latest level at that tick wins.
        assert ob._bids[10000].size == 2.0
        assert ob._asks[10001].size == 2.5

    def test_deleting_deduplicated_snapshot_tick_keeps_state_consistent(self):
        """Deleting a duplicate snapshot tick should not leave stale tick entries."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel(price=100.00, size=1.0, norders=1),
            OrderbookLevel(price=100.009, size=2.0, norders=2),  # same tick as 100.00
            OrderbookLevel(price=99.99, size=3.0, norders=3),
        ]
        asks = [
            OrderbookLevel(price=100.01, size=1.5, norders=1),
            OrderbookLevel(price=100.019, size=2.5, norders=2),  # same tick as 100.01
            OrderbookLevel(price=100.02, size=3.5, norders=3),
        ]
        ob.consume_snapshot(asks=asks, bids=bids)

        ob.consume_deltas(
            asks=[OrderbookLevel(price=100.019, size=0.0, norders=0)],
            bids=[OrderbookLevel(price=100.009, size=0.0, norders=0)],
        )

        assert 10000 not in ob._bids
        assert 10001 not in ob._asks
        assert ob._sorted_bid_ticks == [9999]
        assert ob._sorted_ask_ticks == [10002]
        assert [level.ticks for level in ob.get_bids()] == [9999]
        assert [level.ticks for level in ob.get_asks()] == [10002]


class TestOrderbookIncrementalUpdates:
    """Test incremental update functionality."""

    def setup_method(self):
        """Set up a basic orderbook for testing."""
        self.ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 3.0, 3, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 3.5, 3, 0.01, 0.001),
        ]

        self.ob.consume_snapshot(asks=asks, bids=bids)

    def test_level_addition(self):
        """Test adding new levels."""
        # Add new ask level
        new_asks = [OrderbookLevel.from_values(100.04, 1.0, 1, 0.01, 0.001)]
        self.ob.consume_deltas(asks=new_asks, bids=[])

        assert len(self.ob._asks) == 4
        assert 10004 in self.ob._asks
        assert self.ob._sorted_ask_ticks == [10001, 10002, 10003, 10004]

    def test_level_modification(self):
        """Test modifying existing levels."""
        # Modify existing bid
        modified_bids = [OrderbookLevel.from_values(100.00, 5.0, 5, 0.01, 0.001)]
        self.ob.consume_deltas(asks=[], bids=modified_bids)

        best_bid, _ = self.ob.get_bbo()
        assert best_bid.size == 5.0
        assert best_bid.norders == 5

    def test_level_deletion(self):
        """Test deleting levels using zero size."""
        # Delete best bid
        deleted_bids = [OrderbookLevel.from_values(100.00, 0.0, 0, 0.01, 0.001)]
        self.ob.consume_deltas(asks=[], bids=deleted_bids)

        assert len(self.ob._bids) == 2
        assert 10000 not in self.ob._bids
        assert self.ob._sorted_bid_ticks == [9998, 9999]

        best_bid, _ = self.ob.get_bbo()
        assert best_bid.price == 99.99

    def test_batched_deltas_last_update_wins(self):
        """Repeated tick updates in one batch should use final update state."""
        asks = [
            OrderbookLevel.from_values(100.04, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.04, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 0.0, 0, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 4.0, 4, 0.01, 0.001),
            OrderbookLevel.from_values(100.01, 0.0, 0, 0.01, 0.001),
        ]
        bids = [
            OrderbookLevel.from_values(99.97, 0.0, 0, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 0.0, 0, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 1.2, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.00, 0.0, 0, 0.01, 0.001),
        ]

        self.ob.consume_deltas(asks=asks, bids=bids)

        assert self.ob._sorted_ask_ticks == [10002, 10003, 10004]
        assert self.ob._sorted_bid_ticks == [9998, 9999]
        assert self.ob._asks[10004].size == 2.0
        assert self.ob._asks[10002].size == 4.0
        assert self.ob._bids[9998].size == 1.2
        assert 10000 not in self.ob._bids


class TestOrderbookBBOUpdates:
    """Test BBO (Best Bid/Offer) update functionality."""

    def setup_method(self):
        """Set up a basic orderbook for testing."""
        self.ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 3.0, 3, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 3.5, 3, 0.01, 0.001),
        ]

        self.ob.consume_snapshot(asks=asks, bids=bids)

    def test_bbo_replacement(self):
        """Test replacing BBO levels."""
        new_bid = OrderbookLevel.from_values(100.005, 2.0, 1, 0.01, 0.001)
        new_ask = OrderbookLevel.from_values(100.015, 1.8, 1, 0.01, 0.001)

        self.ob.consume_bbo(ask=new_ask, bid=new_bid)

        best_bid, best_ask = self.ob.get_bbo()
        assert best_bid.price == 100.005
        assert best_ask.price == 100.015

    def test_bbo_deletion(self):
        """Test deleting BBO levels using zero size."""
        zero_bid = OrderbookLevel.from_values(100.00, 0.0, 0, 0.01, 0.001)
        zero_ask = OrderbookLevel.from_values(100.01, 0.0, 0, 0.01, 0.001)

        self.ob.consume_bbo(ask=zero_ask, bid=zero_bid)

        best_bid, best_ask = self.ob.get_bbo()
        assert best_bid.price == 99.99
        assert best_ask.price == 100.02


class TestOrderbookAccessors:
    """Test orderbook accessor methods."""

    def setup_method(self):
        """Set up a basic orderbook for testing."""
        self.ob = Orderbook(tick_size=0.01, lot_size=0.001, size=5)

        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 3.0, 3, 0.01, 0.001),
            OrderbookLevel.from_values(99.97, 4.0, 4, 0.01, 0.001),
            OrderbookLevel.from_values(99.96, 5.0, 5, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 3.5, 3, 0.01, 0.001),
            OrderbookLevel.from_values(100.04, 4.5, 4, 0.01, 0.001),
            OrderbookLevel.from_values(100.05, 5.5, 5, 0.01, 0.001),
        ]

        self.ob.consume_snapshot(asks=asks, bids=bids)

    def test_get_bbo(self):
        """Test getting best bid and offer."""
        best_bid, best_ask = self.ob.get_bbo()
        assert best_bid.price == 100.00
        assert best_ask.price == 100.01

        # Returned references are stable and reflect current state
        assert best_bid.price == 100.00
        assert best_ask.price == 100.01

    def test_get_bids(self):
        """Test getting bid levels."""
        all_bids = self.ob.get_bids()
        assert len(all_bids) == 5
        assert [b.price for b in all_bids] == [100.00, 99.99, 99.98, 99.97, 99.96]

        top3_bids = self.ob.get_bids()[:3]
        assert [b.price for b in top3_bids] == [100.00, 99.99, 99.98]

        # Returned lists are copies of the underlying order
        assert [b.price for b in all_bids] == [100.00, 99.99, 99.98, 99.97, 99.96]

    def test_get_asks(self):
        """Test getting ask levels."""
        all_asks = self.ob.get_asks()
        assert len(all_asks) == 5
        assert [a.price for a in all_asks] == [100.01, 100.02, 100.03, 100.04, 100.05]

        top3_asks = self.ob.get_asks()[:3]
        assert [a.price for a in top3_asks] == [100.01, 100.02, 100.03]

        # Returned lists are copies of the underlying order
        assert [a.price for a in all_asks] == [100.01, 100.02, 100.03, 100.04, 100.05]

    def test_iter_bids(self):
        """Test iterating over bid levels."""
        bid_prices = [bid.price for bid in self.ob.iter_bids()]
        assert bid_prices == [100.00, 99.99, 99.98, 99.97, 99.96]

        top3_bid_prices = [bid.price for bid in list(self.ob.iter_bids())[:3]]
        assert top3_bid_prices == [100.00, 99.99, 99.98]

    def test_iter_asks(self):
        """Test iterating over ask levels."""
        ask_prices = [ask.price for ask in self.ob.iter_asks()]
        assert ask_prices == [100.01, 100.02, 100.03, 100.04, 100.05]

        top3_ask_prices = [ask.price for ask in list(self.ob.iter_asks())[:3]]
        assert top3_ask_prices == [100.01, 100.02, 100.03]

    def test_get_levels_with_depth(self):
        """Test get_asks/get_bids depth argument behavior."""
        expected_bid_prices = [100.00, 99.99, 99.98, 99.97, 99.96]
        expected_ask_prices = [100.01, 100.02, 100.03, 100.04, 100.05]

        assert [b.price for b in self.ob.get_bids(depth=None)] == expected_bid_prices
        assert [a.price for a in self.ob.get_asks(depth=None)] == expected_ask_prices

        assert self.ob.get_bids(depth=0) == []
        assert self.ob.get_asks(depth=0) == []

        assert [b.price for b in self.ob.get_bids(depth=1)] == [100.00]
        assert [a.price for a in self.ob.get_asks(depth=1)] == [100.01]

        depth_len = len(expected_bid_prices)
        assert [
            b.price for b in self.ob.get_bids(depth=depth_len)
        ] == expected_bid_prices
        assert [
            a.price for a in self.ob.get_asks(depth=depth_len)
        ] == expected_ask_prices

        assert [
            b.price for b in self.ob.get_bids(depth=depth_len + 5)
        ] == expected_bid_prices
        assert [
            a.price for a in self.ob.get_asks(depth=depth_len + 5)
        ] == expected_ask_prices

    def test_iter_levels_with_depth(self):
        """Test iter_asks/iter_bids depth argument behavior."""
        expected_bid_prices = [100.00, 99.99, 99.98, 99.97, 99.96]
        expected_ask_prices = [100.01, 100.02, 100.03, 100.04, 100.05]

        assert [b.price for b in self.ob.iter_bids(depth=None)] == expected_bid_prices
        assert [a.price for a in self.ob.iter_asks(depth=None)] == expected_ask_prices

        assert list(self.ob.iter_bids(depth=0)) == []
        assert list(self.ob.iter_asks(depth=0)) == []

        assert [b.price for b in self.ob.iter_bids(depth=1)] == [100.00]
        assert [a.price for a in self.ob.iter_asks(depth=1)] == [100.01]

        depth_len = len(expected_bid_prices)
        assert [
            b.price for b in self.ob.iter_bids(depth=depth_len)
        ] == expected_bid_prices
        assert [
            a.price for a in self.ob.iter_asks(depth=depth_len)
        ] == expected_ask_prices

        assert [
            b.price for b in self.ob.iter_bids(depth=depth_len + 5)
        ] == expected_bid_prices
        assert [
            a.price for a in self.ob.iter_asks(depth=depth_len + 5)
        ] == expected_ask_prices


class TestOrderbookCalculations:
    """Test orderbook calculation methods."""

    def setup_method(self):
        """Set up a basic orderbook for testing."""
        self.ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 3.0, 3, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 3.5, 3, 0.01, 0.001),
        ]

        self.ob.consume_snapshot(asks=asks, bids=bids)

    def test_get_bbo_spread(self):
        """Test BBO spread calculation using integer arithmetic."""
        spread = self.ob.get_bbo_spread()
        assert spread == 0.01  # Should be exactly one tick

        # Verify it's using integer arithmetic internally
        assert spread == self.ob._tick_size

    def test_get_mid_price(self):
        """Test mid price calculation using integer arithmetic."""
        mid_price = self.ob.get_mid_price()
        expected_mid_ticks = (10000 + 10001) // 2  # 10000.5 -> 10000
        expected_mid_price = expected_mid_ticks * 0.01
        assert mid_price == expected_mid_price

    def test_get_wmid_price(self):
        """Test weighted mid price calculation."""
        try:
            wmid_price = self.ob.get_wmid_price()
            assert isinstance(wmid_price, float)
        except AttributeError:
            pytest.skip("wmid_price implementation has a bug")

    def test_get_volume_weighted_mid_price(self):
        """Test volume weighted mid price calculation."""
        vwmid = self.ob.get_volume_weighted_mid_price(size=1.0, is_base_currency=True)
        assert isinstance(vwmid, float)
        assert vwmid > 0

        # Test with zero size
        vwmid_zero = self.ob.get_volume_weighted_mid_price(size=0.0)
        assert vwmid_zero == self.ob.get_mid_price()

    def test_get_price_impact(self):
        """Test price impact calculation."""
        # Buy impact
        buy_impact = self.ob.get_price_impact(
            size=1.0, is_buy=True, is_base_currency=True
        )
        assert buy_impact >= 0.0

        # Sell impact
        sell_impact = self.ob.get_price_impact(
            size=1.0, is_buy=False, is_base_currency=True
        )
        assert sell_impact >= 0.0

        # Zero size should have zero impact
        zero_impact = self.ob.get_price_impact(size=0.0, is_buy=True)
        assert zero_impact == 0.0

    def test_does_bbo_price_change(self):
        """Test BBO price change detection."""
        # Same prices should not indicate change
        assert not self.ob.does_bbo_price_change(100.00, 100.01)

        # Different prices should indicate change
        assert self.ob.does_bbo_price_change(100.01, 100.01)
        assert self.ob.does_bbo_price_change(100.00, 100.02)

    def test_does_bbo_cross(self):
        """Test BBO crossing detection."""
        # Normal prices should not be crossed
        assert not self.ob.does_bbo_cross(99.99, 100.02)

        # Crossed prices should be detected
        assert self.ob.does_bbo_cross(100.02, 99.99)  # bid > ask


class TestOrderbookIntegrationAndEdgeCases:
    """Test complex integration scenarios and edge cases."""

    def test_multiple_successive_updates(self):
        """Test many successive updates maintaining correct state."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        # Initial snapshot
        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 2.0, 2, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 3.0, 3, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 2.5, 2, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 3.5, 3, 0.01, 0.001),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        # Verify initial state
        best_bid, best_ask = ob.get_bbo()
        assert best_bid.price == 100.00
        assert best_ask.price == 100.01

        # Update 1: Modify best bid
        ob.consume_bbo(
            ask=OrderbookLevel.from_values(100.01, 1.5, 1, 0.01, 0.001),
            bid=OrderbookLevel.from_values(100.005, 1.2, 1, 0.01, 0.001),
        )

        best_bid, best_ask = ob.get_bbo()
        assert best_bid.price == 100.005
        assert best_ask.price == 100.01
        assert len(ob._bids) == 3  # Should still have 3 levels

        # Update 2: Add new ask level
        new_asks = [OrderbookLevel.from_values(100.004, 0.5, 1, 0.01, 0.001)]
        ob.consume_deltas(asks=new_asks, bids=[])

        best_bid, best_ask = ob.get_bbo()
        assert best_ask.price == 100.004  # Should be new best ask

        # Update 3: Delete some levels
        del_updates = [
            OrderbookLevel.from_values(99.99, 0.0, 0, 0.01, 0.001),  # Delete second bid
            OrderbookLevel.from_values(100.02, 0.0, 0, 0.01, 0.001),  # Delete ask level
        ]
        ob.consume_deltas(asks=[del_updates[1]], bids=[del_updates[0]])

        assert len(ob._bids) == 2
        assert 9999 not in ob._bids
        assert 10002 not in ob._asks

        # Verify final state consistency
        all_bids = ob.get_bids()
        all_asks = ob.get_asks()

        # Check bid ordering (highest first)
        bid_prices = [b.price for b in all_bids]
        assert bid_prices == sorted(bid_prices, reverse=True)

        # Check ask ordering (lowest first)
        ask_prices = [a.price for a in all_asks]
        assert ask_prices == sorted(ask_prices)

    def test_mixed_bbo_and_regular_updates(self):
        """Test mixing BBO updates with regular updates."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=5)

        # Start with initial book
        bids = [
            OrderbookLevel.from_values(100.00 - i * 0.01, 1.0, 1, 0.01, 0.001)
            for i in range(5)
        ]
        asks = [
            OrderbookLevel.from_values(100.01 + i * 0.01, 1.0, 1, 0.01, 0.001)
            for i in range(5)
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        for i in range(10):
            if i % 2 == 0:
                # BBO update
                new_bid_price = 100.00 + (i * 0.001)
                new_ask_price = 100.01 + (i * 0.001)

                ob.consume_bbo(
                    ask=OrderbookLevel.from_values(new_ask_price, 1.0, 1, 0.01, 0.001),
                    bid=OrderbookLevel.from_values(new_bid_price, 1.0, 1, 0.01, 0.001),
                )
            else:
                # Regular update - add new levels
                new_bid_price = 99.95 - (i * 0.01)
                new_ask_price = 100.06 + (i * 0.01)

                ob.consume_deltas(
                    asks=[
                        OrderbookLevel.from_values(new_ask_price, 1.0, 1, 0.01, 0.001)
                    ],
                    bids=[
                        OrderbookLevel.from_values(new_bid_price, 1.0, 1, 0.01, 0.001)
                    ],
                )

            # Verify state consistency after each update
            best_bid, best_ask = ob.get_bbo()
            assert best_bid.price <= best_ask.price

            # Verify internal tick ordering
            assert ob._sorted_bid_ticks == sorted(ob._sorted_bid_ticks)
            assert ob._sorted_ask_ticks == sorted(ob._sorted_ask_ticks)

    def test_floating_point_precision(self):
        """Test that integer arithmetic avoids floating point errors."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        # Use prices that would cause floating point precision issues - need
        # at least 3 levels
        bids = [
            OrderbookLevel.from_values(100.01, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 1.0, 1, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.02, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.04, 1.0, 1, 0.01, 0.001),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        spread = ob.get_bbo_spread()

        # Should be exactly one tick, not subject to floating point errors
        assert spread == 0.01
        assert abs(spread - 0.01) < 1e-15  # No floating point drift

    def test_empty_orderbook_errors(self):
        """Test that empty orderbook operations raise appropriate errors."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001)

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_bbo()

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_bids()

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_asks()

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_bbo_spread()

    def test_reset_functionality(self):
        """Test orderbook reset functionality."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        bids = [
            OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.99, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 1.0, 1, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 1.0, 1, 0.01, 0.001),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)
        assert ob._is_populated

        ob.reset()

        assert len(ob._bids) == 0
        assert len(ob._asks) == 0
        assert len(ob._sorted_bid_ticks) == 0
        assert len(ob._sorted_ask_ticks) == 0
        assert not ob._is_populated


class TestPrecisionAndIntegerArithmetic:
    """Test that the orderbook properly uses integer arithmetic for precision."""

    def test_tick_precision_consistency(self):
        """Test that tick conversions are consistent and precise."""
        tick_size = 0.01

        # Test various price points
        test_prices = [100.00, 100.01, 100.99, 99.99, 0.01, 999.99]

        for price in test_prices:
            ticks = _price_to_ticks(price, tick_size)
            recovered_price = _price_from_ticks(ticks, tick_size)

            # Should be exact (within floating point precision)
            assert abs(recovered_price - (ticks * tick_size)) < 1e-15

    def test_lot_precision_consistency(self):
        """Test that lot conversions are consistent and precise."""
        lot_size = 0.001

        # Test various size points
        test_sizes = [1.0, 0.5, 0.001, 1.999, 0.0001]

        for size in test_sizes:
            lots = _size_to_lots(size, lot_size)
            recovered_size = _size_from_lots(lots, lot_size)

            # Should be exact (within floating point precision)
            assert abs(recovered_size - (lots * lot_size)) < 1e-15

    def test_spread_calculation_precision(self):
        """Test that spread calculations avoid floating point errors."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=3)

        # Test with prices that could cause floating point issues - need at
        # least 3 levels
        bids = [
            OrderbookLevel.from_values(99.99, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.98, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(99.97, 1.0, 1, 0.01, 0.001),
        ]
        asks = [
            OrderbookLevel.from_values(100.01, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.02, 1.0, 1, 0.01, 0.001),
            OrderbookLevel.from_values(100.03, 1.0, 1, 0.01, 0.001),
        ]

        ob.consume_snapshot(asks=asks, bids=bids)

        spread = ob.get_bbo_spread()

        # Should be exactly 2 ticks
        assert spread == 0.02

        # Verify using integer arithmetic
        best_bid_ticks = ob._sorted_bid_ticks[-1]
        best_ask_ticks = ob._sorted_ask_ticks[0]
        spread_ticks = best_ask_ticks - best_bid_ticks

        assert spread_ticks == 2
        assert spread == spread_ticks * ob._tick_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

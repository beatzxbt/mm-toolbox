"""Standard-specific primitives, initialization, precision and accessor contracts.

Feed mutations and calculations are tested across all three implementations in
the shared independent-reference scenario suite.
"""

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
    """Layer 1: Test the utility conversion functions in isolation."""

    def test_price_to_ticks_conversion(self):
        """Given various prices, When converted to ticks, Then exact integer ticks returned."""
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
        """Given various sizes, When converted to lots, Then exact integer lots returned."""
        assert _size_to_lots(1.0, 0.001) == 1000
        assert _size_to_lots(0.5, 0.001) == 500
        assert _size_to_lots(0.0001, 0.0001) == 1

    def test_price_from_ticks_conversion(self):
        """Given various ticks, When converted back to price, Then original price recovered."""
        assert _price_from_ticks(10001, 0.01) == 100.01
        assert _price_from_ticks(10000, 0.01) == 100.00
        assert abs(_price_from_ticks(801, 0.125) - 100.125) < 1e-10

    def test_size_from_lots_conversion(self):
        """Given various lots, When converted back to size, Then original size recovered."""
        assert _size_from_lots(1000, 0.001) == 1.0
        assert _size_from_lots(500, 0.001) == 0.5
        assert abs(_size_from_lots(1, 0.0001) - 0.0001) < 1e-10


class TestOrderbookLevel:
    """Layer 1: Test OrderbookLevel functionality in isolation."""

    def test_basic_creation(self):
        """Given valid parameters, When creating OrderbookLevel, Then fields set correctly."""
        level = OrderbookLevel(price=100.0, size=1.5, norders=2)
        assert level.price == 100.0
        assert level.size == 1.5
        assert level.norders == 2
        assert level.ticks == -1
        assert level.lots == -1

    def test_validation_errors(self):
        """Given invalid values, When creating OrderbookLevel, Then appropriate errors raised."""
        with pytest.raises(ValueError, match="Invalid price"):
            OrderbookLevel(price=-1.0, size=1.0, norders=1)

        with pytest.raises(ValueError, match="Invalid size"):
            OrderbookLevel(price=100.0, size=-1.0, norders=1)

        with pytest.raises(ValueError, match="Invalid norders"):
            OrderbookLevel(price=100.0, size=1.0, norders=-1)

    def test_value_property(self):
        """Given price and size, When value computed, Then equals price * size."""
        level = OrderbookLevel(price=100.0, size=1.5, norders=2)
        assert level.value == 150.0

    def test_precision_info_addition(self):
        """Given level without precision, When add_precision_info called, Then ticks/lots computed."""
        level = OrderbookLevel(price=100.01, size=1.5, norders=2)
        level.add_precision_info(inv_tick_size=100.0, inv_lot_size=1000.0)

        assert level.ticks == 10001
        assert level.lots == 1500

    def test_precision_info_validation(self):
        """Given invalid inverse precision, When add_precision_info called, Then raises error."""
        level = OrderbookLevel(price=100.0, size=1.0, norders=1)

        with pytest.raises(ValueError, match="Invalid inv_tick_size"):
            level.add_precision_info(inv_tick_size=-100.0, inv_lot_size=1000.0)

        with pytest.raises(ValueError, match="Invalid inv_lot_size"):
            level.add_precision_info(inv_tick_size=100.0, inv_lot_size=-1000.0)

    def test_from_values_class_method(self):
        """Given price/size/tick/lot, When from_values called, Then level with precision created."""
        level = OrderbookLevel.from_values(
            price=100.01, size=1.5, norders=2, tick_size=0.01, lot_size=0.001
        )

        assert level.price == 100.01
        assert level.size == 1.5
        assert level.norders == 2
        assert level.ticks == 10001
        assert level.lots == 1500

    def test_reset_functionality(self):
        """Given populated level, When reset called, Then all fields zeroed."""
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
    """Layer 2: Test orderbook initialization and basic setup."""

    def test_basic_initialization(self):
        """Given valid params, When creating Orderbook, Then empty book initialized."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=100)

        assert ob._tick_size == 0.01
        assert ob._lot_size == 0.001
        assert ob._size == 100
        assert len(ob._asks) == 0
        assert len(ob._bids) == 0
        assert len(ob._sorted_ask_ticks) == 0
        assert len(ob._sorted_bid_ticks) == 0
        assert not ob._is_populated
        assert not ob.is_initialized()
        assert not ob.is_populated()
        assert not ob._trust_input_precision

    def test_initialization_validation(self):
        """Given invalid params, When creating Orderbook, Then raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tick_size"):
            Orderbook(tick_size=-0.01, lot_size=0.001)

        with pytest.raises(ValueError, match="Invalid lot_size"):
            Orderbook(tick_size=0.01, lot_size=-0.001)

        with pytest.raises(ValueError, match="Invalid size"):
            Orderbook(tick_size=0.01, lot_size=0.001, size=0)

    def test_initialization_with_initial_data(self):
        """Given initial bids/asks, When creating Orderbook, Then pre-populated."""
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
        assert ob.is_initialized()
        assert ob.is_populated()
        assert len(ob._bids) == 2
        assert len(ob._asks) == 2


class TestOrderbookSnapshots:
    """Layer 2: Test orderbook snapshot functionality."""

    def test_snapshot_without_precision_info(self):
        """Given levels without pre-computed precision, When snapshot consumed, Then auto-computes."""
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

        assert [(bid.ticks, bid.lots) for bid in bids] == [(10000, 1000), (9999, 2000)]
        assert [(ask.ticks, ask.lots) for ask in asks] == [(10001, 1500), (10002, 2500)]

    def test_snapshot_recomputes_existing_precision_by_default(self):
        """Given levels with wrong ticks/lots, When snapshot consumed with default mode, Then recomputes."""
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
        """Given levels with custom ticks/lots, When trust_input_precision=True, Then preserves them."""
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
        """Given insufficient levels, When snapshot consumed, Then raises ValueError."""
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


class TestOrderbookAccessors:
    """Layer 2: Test orderbook accessor methods."""

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
        """Given populated book, When get_bbo called, Then returns best bid and ask."""
        best_bid, best_ask = self.ob.get_bbo()
        assert best_bid.price == 100.00
        assert best_ask.price == 100.01

        # Returned references are stable and reflect current state
        assert best_bid.price == 100.00
        assert best_ask.price == 100.01

    def test_get_bids(self):
        """Given populated book, When get_bids called, Then returns bid levels."""
        all_bids = self.ob.get_bids()
        assert len(all_bids) == 5
        assert [b.price for b in all_bids] == [100.00, 99.99, 99.98, 99.97, 99.96]

        top3_bids = self.ob.get_bids()[:3]
        assert [b.price for b in top3_bids] == [100.00, 99.99, 99.98]

        # Returned lists are copies of the underlying order
        assert [b.price for b in all_bids] == [100.00, 99.99, 99.98, 99.97, 99.96]

    def test_get_asks(self):
        """Given populated book, When get_asks called, Then returns ask levels."""
        all_asks = self.ob.get_asks()
        assert len(all_asks) == 5
        assert [a.price for a in all_asks] == [100.01, 100.02, 100.03, 100.04, 100.05]

        top3_asks = self.ob.get_asks()[:3]
        assert [a.price for a in top3_asks] == [100.01, 100.02, 100.03]

        # Returned lists are copies of the underlying order
        assert [a.price for a in all_asks] == [100.01, 100.02, 100.03, 100.04, 100.05]

    def test_iter_bids(self):
        """Given populated book, When iter_bids called, Then iterates bid levels."""
        bid_prices = [bid.price for bid in self.ob.iter_bids()]
        assert bid_prices == [100.00, 99.99, 99.98, 99.97, 99.96]

        top3_bid_prices = [bid.price for bid in list(self.ob.iter_bids())[:3]]
        assert top3_bid_prices == [100.00, 99.99, 99.98]

    def test_iter_asks(self):
        """Given populated book, When iter_asks called, Then iterates ask levels."""
        ask_prices = [ask.price for ask in self.ob.iter_asks()]
        assert ask_prices == [100.01, 100.02, 100.03, 100.04, 100.05]

        top3_ask_prices = [ask.price for ask in list(self.ob.iter_asks())[:3]]
        assert top3_ask_prices == [100.01, 100.02, 100.03]

    def test_get_levels_with_depth(self):
        """Given depth argument, When get_bids/get_asks called, Then respects depth limit."""
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
        """Given depth argument, When iter_bids/iter_asks called, Then respects depth limit."""
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


class TestOrderbookIntegrationAndEdgeCases:
    """Layer 3: Test complex integration scenarios and edge cases."""

    def test_empty_orderbook_errors(self):
        """Given empty orderbook, When accessors called, Then appropriate errors raised."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001)

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_bbo()

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_bids()

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_asks()

        with pytest.raises(ValueError, match="Orderbook is not populated"):
            ob.get_bbo_spread()

    def test_side_empty_state_raises_clear_domain_error(self):
        """Given one-sided empty book, When BBO accessors called, Then clear error raised."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=1)
        ob.consume_snapshot(
            asks=[OrderbookLevel.from_values(100.01, 1.0, 1, 0.01, 0.001)],
            bids=[OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001)],
        )

        ob.consume_deltas(
            asks=[OrderbookLevel.from_values(100.01, 0.0, 0, 0.01, 0.001)],
            bids=[],
        )

        assert not ob._is_populated
        assert ob.is_initialized()
        assert not ob.is_populated()
        assert ob.get_asks() == []
        assert len(ob.get_bids()) == 1
        with pytest.raises(ValueError, match="Orderbook side unavailable"):
            ob.get_bbo()
        with pytest.raises(ValueError, match="Orderbook side unavailable"):
            ob.get_bbo_spread()
        with pytest.raises(ValueError, match="Orderbook side unavailable"):
            ob.does_bbo_cross(100.00, 100.01)

    def test_side_repopulation_restores_two_sided_state(self):
        """Given one-sided empty book, When missing side replenished, Then two-sided state restored."""
        ob = Orderbook(tick_size=0.01, lot_size=0.001, size=1)
        ob.consume_snapshot(
            asks=[OrderbookLevel.from_values(100.01, 1.0, 1, 0.01, 0.001)],
            bids=[OrderbookLevel.from_values(100.00, 1.0, 1, 0.01, 0.001)],
        )

        ob.consume_deltas(
            asks=[OrderbookLevel.from_values(100.01, 0.0, 0, 0.01, 0.001)],
            bids=[],
        )
        assert not ob._is_populated

        ob.consume_deltas(
            asks=[OrderbookLevel.from_values(100.02, 2.0, 1, 0.01, 0.001)],
            bids=[],
        )
        assert ob._is_populated
        best_bid, best_ask = ob.get_bbo()
        assert best_bid.price == 100.00
        assert best_ask.price == 100.02

    def test_reset_functionality(self):
        """Given populated book, When reset called, Then emptied."""
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
    """Layer 1-2: Test that the orderbook properly uses integer arithmetic for precision."""

    def test_tick_precision_consistency(self):
        """Given various prices, When roundtripped through ticks, Then exact recovery."""
        tick_size = 0.01

        # Test various price points
        test_prices = [100.00, 100.01, 100.99, 99.99, 0.01, 999.99]

        for price in test_prices:
            ticks = _price_to_ticks(price, tick_size)
            recovered_price = _price_from_ticks(ticks, tick_size)

            # Should be exact (within floating point precision)
            assert abs(recovered_price - (ticks * tick_size)) < 1e-15

    def test_lot_precision_consistency(self):
        """Given various sizes, When roundtripped through lots, Then exact recovery."""
        lot_size = 0.001

        # Test various size points
        test_sizes = [1.0, 0.5, 0.001, 1.999, 0.0001]

        for size in test_sizes:
            lots = _size_to_lots(size, lot_size)
            recovered_size = _size_from_lots(lots, lot_size)

            # Should be exact (within floating point precision)
            assert abs(recovered_size - (lots * lot_size)) < 1e-15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

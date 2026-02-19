"""Comprehensive tests for the Python-facing advanced orderbook."""

from __future__ import annotations

import numpy as np
import pytest

from mm_toolbox.orderbook.advanced import (
    AdvancedOrderbook,
    OrderbookLevel,
    OrderbookLevels,
    PyOrderbookSortedness,
    convert_price_from_tick,
    convert_price_to_tick,
    convert_size_from_lot,
    convert_size_to_lot,
)

# Aliases for backward compatibility in test code
PyAdvancedOrderbook = AdvancedOrderbook
PyOrderbookLevel = OrderbookLevel
PyOrderbookLevels = OrderbookLevels

TICK_SIZE = 0.01
LOT_SIZE = 0.001


def _make_levels(
    prices: list[float],
    sizes: list[float],
    norders: list[int] | None = None,
    *,
    with_precision: bool,
) -> tuple[PyOrderbookLevels, bool]:
    norders = [1] * len(prices) if norders is None else norders
    if with_precision:
        return (
            PyOrderbookLevels.from_list_with_ticks_and_lots(
                prices, sizes, norders, TICK_SIZE, LOT_SIZE
            ),
            True,
        )
    return PyOrderbookLevels.from_list(prices, sizes, norders), False


def _empty_levels() -> PyOrderbookLevels:
    """A no-op delta payload for asks that won't affect a typical test orderbook.

    Uses price=999.0 which is far ABOVE the test orderbook asks (~100.0).
    With size=0 and being outside the ask range, this level will be ignored.
    """
    return PyOrderbookLevels.from_list_with_ticks_and_lots(
        [999.0], [0.0], [0], TICK_SIZE, LOT_SIZE
    )


def _empty_bid_levels() -> PyOrderbookLevels:
    """A no-op delta payload for bids that won't affect a typical test orderbook.

    Uses price=1.0 which is far BELOW the test orderbook bids (~100.0).
    With size=0 and being outside the bid range, this level will be ignored.
    """
    return PyOrderbookLevels.from_list_with_ticks_and_lots(
        [1.0], [0.0], [0], TICK_SIZE, LOT_SIZE
    )


def _mk_book(num_levels: int = 8) -> PyAdvancedOrderbook:
    return PyAdvancedOrderbook(
        tick_size=TICK_SIZE,
        lot_size=LOT_SIZE,
        num_levels=num_levels,
        delta_sortedness=PyOrderbookSortedness.UNKNOWN,
        snapshot_sortedness=PyOrderbookSortedness.UNKNOWN,
    )


def _bids_asks_arrays(book: PyAdvancedOrderbook) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(book.get_bids_numpy()), np.asarray(book.get_asks_numpy())


def _bbo_prices(book: PyAdvancedOrderbook) -> tuple[float, float]:
    bids, asks = _bids_asks_arrays(book)
    return float(bids["price"][0]), float(asks["price"][0])


class TestUtilityFunctions:
    """Test the utility conversion helpers."""

    def test_price_to_ticks_conversion(self):
        assert convert_price_to_tick(100.01, TICK_SIZE) == 10001
        assert convert_price_to_tick(100.0, TICK_SIZE) == 10000
        assert convert_price_to_tick(99.99, TICK_SIZE) == 9999
        assert convert_price_to_tick(100.125, 0.125) == 801
        assert convert_price_to_tick(100.0001, 0.0001) == 1000001

    def test_size_to_lots_conversion(self):
        assert convert_size_to_lot(1.0, LOT_SIZE) == 1000
        assert convert_size_to_lot(0.5, LOT_SIZE) == 500
        assert convert_size_to_lot(0.0001, 0.0001) == 1

    def test_price_from_ticks_conversion(self):
        assert convert_price_from_tick(10001, TICK_SIZE) == 100.01
        assert convert_price_from_tick(10000, TICK_SIZE) == 100.0
        assert abs(convert_price_from_tick(801, 0.125) - 100.125) < 1e-10

    def test_size_from_lots_conversion(self):
        assert convert_size_from_lot(1000, LOT_SIZE) == 1.0
        assert convert_size_from_lot(500, LOT_SIZE) == 0.5
        assert abs(convert_size_from_lot(1, 0.0001) - 0.0001) < 1e-10


class TestPyOrderbookLevel:
    """Validate PyOrderbookLevel creation and helpers."""

    def test_basic_creation_and_validation(self):
        level = PyOrderbookLevel(price=100.0, size=1.5, norders=2, verify_values=True)
        assert level.price == pytest.approx(100.0)
        assert level.size == pytest.approx(1.5)
        assert level.norders == 2
        assert level.ticks == 0
        assert level.lots == 0

    def test_invalid_values_raise(self):
        with pytest.raises(ValueError):
            PyOrderbookLevel(price=-1.0, size=1.0, norders=1)
        with pytest.raises(ValueError):
            PyOrderbookLevel(price=1.0, size=-1.0, norders=1)
        # norders is u64, so negative values cause OverflowError at the Cython type conversion level
        with pytest.raises((ValueError, OverflowError)):
            PyOrderbookLevel(price=1.0, size=1.0, norders=-1)

    def test_with_ticks_and_lots(self):
        level = PyOrderbookLevel.with_ticks_and_lots(
            price=100.01, size=1.5, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=2
        )
        assert level.ticks == convert_price_to_tick(100.01, TICK_SIZE)
        assert level.lots == convert_size_to_lot(1.5, LOT_SIZE)


class TestInitialization:
    """Test orderbook initialization and guards."""

    def test_basic_initialization(self):
        ob = _mk_book(num_levels=64)
        # Empty calls should raise
        with pytest.raises(RuntimeError):
            ob.get_mid_price()

    def test_invalid_init_parameters(self):
        with pytest.raises(ValueError):
            PyAdvancedOrderbook(tick_size=-0.01, lot_size=LOT_SIZE, num_levels=64)
        with pytest.raises(ValueError):
            PyAdvancedOrderbook(tick_size=TICK_SIZE, lot_size=-0.001, num_levels=64)


class TestSnapshots:
    """Test snapshot ingestion and sorting."""

    def test_basic_snapshot_update(self):
        ob = _mk_book(num_levels=64)
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

        ob.consume_snapshot(asks, bids)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == [100.0, 99.99, 99.98]
        assert list(ask_arr["price"]) == [100.01, 100.02, 100.03]
        assert ob.get_bbo_spread() == pytest.approx(0.01)

    def test_snapshot_without_precision(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.0, 99.99],
            sizes=[1.0, 2.0],
            norders=[1, 2],
            with_precision=False,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02],
            sizes=[1.5, 2.5],
            norders=[1, 2],
            with_precision=False,
        )

        ob.consume_snapshot(asks, bids)
        bid_arr, ask_arr = _bids_asks_arrays(ob)

        assert bid_arr["ticks"][0] == convert_price_to_tick(100.0, TICK_SIZE)
        assert ask_arr["ticks"][0] == convert_price_to_tick(100.01, TICK_SIZE)
        # Mid price uses integer tick arithmetic: (10000 + 10001) // 2 = 10000 ticks = 100.0
        expected_mid_ticks = (
            convert_price_to_tick(100.0, TICK_SIZE)
            + convert_price_to_tick(100.01, TICK_SIZE)
        ) // 2
        assert ob.get_mid_price() == convert_price_from_tick(
            expected_mid_ticks, TICK_SIZE
        )


class TestIncrementalUpdates:
    """Test deltas on the advanced orderbook."""

    def setup_method(self):
        self.ob = _mk_book(num_levels=64)
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
        self.ob.consume_snapshot(asks, bids)

    def test_add_new_level(self):
        new_asks, _ = _make_levels(
            prices=[100.04], sizes=[1.0], norders=[1], with_precision=True
        )
        empty_bids = _empty_bid_levels()

        self.ob.consume_deltas(new_asks, empty_bids)
        _, ask_arr = _bids_asks_arrays(self.ob)
        assert list(ask_arr["price"]) == [100.01, 100.02, 100.03, 100.04]

    def test_modify_existing_level(self):
        mod_bids, _ = _make_levels(
            prices=[100.0], sizes=[5.0], norders=[5], with_precision=True
        )
        empty = _empty_levels()

        self.ob.consume_deltas(empty, mod_bids)
        bid_arr, _ = _bids_asks_arrays(self.ob)
        assert bid_arr["size"][0] == pytest.approx(5.0)
        assert bid_arr["norders"][0] == 5

    def test_delete_level_with_zero_size(self):
        del_bids, _ = _make_levels(
            prices=[100.0], sizes=[0.0], norders=[0], with_precision=True
        )
        empty = _empty_levels()

        self.ob.consume_deltas(empty, del_bids)
        bid_arr, _ = _bids_asks_arrays(self.ob)
        assert list(bid_arr["price"]) == [99.99, 99.98]


class TestBBOUpdates:
    """Test BBO-specific updates."""

    def setup_method(self):
        self.ob = _mk_book(num_levels=64)
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
        self.ob.consume_snapshot(asks, bids)

    def test_bbo_replacement(self):
        # Use prices that land on different tick boundaries (tick_size=0.01)
        # 100.01 is a higher bid than current best of 100.0
        # 100.02 is a higher ask than current best of 100.01
        new_bid = PyOrderbookLevel.with_ticks_and_lots(
            price=100.01, size=2.0, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=1
        )
        new_ask = PyOrderbookLevel.with_ticks_and_lots(
            price=100.02, size=1.8, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=1
        )

        self.ob.consume_bbo(new_ask, new_bid)

        bid_price, ask_price = _bbo_prices(self.ob)
        assert bid_price == pytest.approx(100.01)
        assert ask_price == pytest.approx(100.02)

    def test_bbo_deletion(self):
        zero_bid = PyOrderbookLevel.with_ticks_and_lots(
            price=100.0, size=0.0, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=0
        )
        zero_ask = PyOrderbookLevel.with_ticks_and_lots(
            price=100.01, size=0.0, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=0
        )

        self.ob.consume_bbo(zero_ask, zero_bid)
        bid_price, ask_price = _bbo_prices(self.ob)
        assert bid_price == pytest.approx(99.99)
        assert ask_price == pytest.approx(100.02)

    def test_bbo_ignored_when_empty(self):
        empty_ob = _mk_book(num_levels=64)
        new_bid = PyOrderbookLevel.with_ticks_and_lots(
            price=100.01, size=1.0, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=1
        )
        new_ask = PyOrderbookLevel.with_ticks_and_lots(
            price=100.02, size=1.0, tick_size=TICK_SIZE, lot_size=LOT_SIZE, norders=1
        )

        empty_ob.consume_bbo(new_ask, new_bid)

        with pytest.raises(RuntimeError):
            empty_ob.get_bbo()


class TestAccessors:
    """Test accessor helpers via numpy views."""

    def setup_method(self):
        self.ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.00, 99.99, 99.98, 99.97, 99.96],
            sizes=[1.0, 2.0, 3.0, 4.0, 5.0],
            norders=[1, 2, 3, 4, 5],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03, 100.04, 100.05],
            sizes=[1.5, 2.5, 3.5, 4.5, 5.5],
            norders=[1, 2, 3, 4, 5],
            with_precision=True,
        )
        self.ob.consume_snapshot(asks, bids)

    def test_get_bbo(self):
        bid, ask = _bbo_prices(self.ob)
        assert bid == pytest.approx(100.00)
        assert ask == pytest.approx(100.01)

    def test_get_bids_and_asks(self):
        bid_arr, ask_arr = _bids_asks_arrays(self.ob)
        assert list(bid_arr["price"]) == [100.00, 99.99, 99.98, 99.97, 99.96]
        assert list(ask_arr["price"]) == [100.01, 100.02, 100.03, 100.04, 100.05]

        assert list(bid_arr["price"][:3]) == [100.00, 99.99, 99.98]
        assert list(ask_arr["price"][:3]) == [100.01, 100.02, 100.03]


class TestCalculations:
    """Test price calculations on the advanced orderbook."""

    def setup_method(self):
        self.ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.00, 99.99, 99.98],
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
        self.ob.consume_snapshot(asks, bids)

    def test_get_bbo_spread(self):
        spread = self.ob.get_bbo_spread()
        assert spread == pytest.approx(0.01)
        assert spread == TICK_SIZE

    def test_get_mid_price(self):
        mid_price = self.ob.get_mid_price()
        expected_mid_ticks = (
            convert_price_to_tick(100.01, TICK_SIZE)
            + convert_price_to_tick(100.0, TICK_SIZE)
        ) // 2
        assert mid_price == convert_price_from_tick(expected_mid_ticks, TICK_SIZE)

    def test_get_wmid_price(self):
        wmid_price = self.ob.get_wmid_price()
        assert isinstance(wmid_price, float)

    def test_get_volume_weighted_mid_price(self):
        vwmid = self.ob.get_volume_weighted_mid_price(size=1.0, is_base_currency=True)
        assert isinstance(vwmid, float)
        assert vwmid > 0

        vwmid_zero = self.ob.get_volume_weighted_mid_price(
            size=0.0, is_base_currency=True
        )
        assert vwmid_zero == self.ob.get_mid_price()

    def test_get_price_impact(self):
        buy_touch_only = self.ob.get_price_impact(
            size=1.0, is_buy=True, is_base_currency=True
        )
        sell_touch_only = self.ob.get_price_impact(
            size=1.0, is_buy=False, is_base_currency=True
        )
        assert buy_touch_only == pytest.approx(0.0)
        assert sell_touch_only == pytest.approx(0.0)

        buy_multi_level = self.ob.get_price_impact(
            size=2.0, is_buy=True, is_base_currency=True
        )
        sell_multi_level = self.ob.get_price_impact(
            size=2.0, is_buy=False, is_base_currency=True
        )
        assert buy_multi_level == pytest.approx(0.01)
        assert sell_multi_level == pytest.approx(0.01)

        zero_impact = self.ob.get_price_impact(
            size=0.0, is_buy=True, is_base_currency=True
        )
        assert zero_impact == 0.0

    def test_get_price_impact_quote_size_uses_touch_anchor(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.0, 99.0],
            sizes=[1.0, 1.0],
            norders=[1, 1],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[200.0, 201.0],
            sizes=[0.7, 1.0],
            norders=[1, 1],
            with_precision=True,
        )
        ob.consume_snapshot(asks, bids)

        buy_impact_quote = ob.get_price_impact(
            size=140.0, is_buy=True, is_base_currency=False
        )
        assert buy_impact_quote == pytest.approx(0.0)

    def test_get_price_impact_insufficient_liquidity_returns_infinity_marker(self):
        buy_impact = self.ob.get_price_impact(size=8.0, is_buy=True, is_base_currency=True)
        assert buy_impact == np.finfo(float).max

    def test_get_size_for_price_impact_bps(self):
        buy_base = self.ob.get_size_for_price_impact_bps(
            impact_bps=1.0, is_buy=True, is_base_currency=True
        )
        assert buy_base == pytest.approx(4.0)

        sell_base = self.ob.get_size_for_price_impact_bps(
            impact_bps=1.0, is_buy=False, is_base_currency=True
        )
        assert sell_base == pytest.approx(3.0)

        buy_quote = self.ob.get_size_for_price_impact_bps(
            impact_bps=2.0, is_buy=True, is_base_currency=False
        )
        assert buy_quote == pytest.approx(750.17)

        zero_bps = self.ob.get_size_for_price_impact_bps(
            impact_bps=0.0, is_buy=True, is_base_currency=True
        )
        assert zero_bps == 0.0

        negative_bps = self.ob.get_size_for_price_impact_bps(
            impact_bps=-1.0, is_buy=False, is_base_currency=True
        )
        assert negative_bps == 0.0

    def test_get_size_for_price_impact_bps_includes_boundary_levels(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[99.0, 98.01],
            sizes=[1.0, 2.0],
            norders=[1, 1],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.0, 101.0],
            sizes=[1.5, 2.5],
            norders=[1, 1],
            with_precision=True,
        )
        ob.consume_snapshot(asks, bids)

        buy_base = ob.get_size_for_price_impact_bps(
            impact_bps=100.0, is_buy=True, is_base_currency=True
        )
        sell_base = ob.get_size_for_price_impact_bps(
            impact_bps=100.0, is_buy=False, is_base_currency=True
        )
        assert buy_base == pytest.approx(4.0)
        assert sell_base == pytest.approx(3.0)

    def test_does_bbo_price_change(self):
        assert not self.ob.does_bbo_price_change(100.00, 100.01)
        assert self.ob.does_bbo_price_change(100.01, 100.01)
        assert self.ob.does_bbo_price_change(100.00, 100.02)

    def test_is_bbo_crossed(self):
        assert not self.ob.is_bbo_crossed(99.99, 100.02)
        assert self.ob.is_bbo_crossed(100.02, 99.99)


class TestIntegrationAndEdgeCases:
    """Integration-style scenarios mirroring the standard suite."""

    def test_multiple_successive_updates(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.00, 99.99, 99.98],
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
        ob.consume_snapshot(asks, bids)

        best_bid, best_ask = _bbo_prices(ob)
        assert best_bid == pytest.approx(100.00)
        assert best_ask == pytest.approx(100.01)

        # Add a tighter bid via deltas
        tighter_bid, _ = _make_levels(
            prices=[100.005],
            sizes=[1.5],
            norders=[1],
            with_precision=True,  # 100.005 -> 10000 ticks
        )
        empty_asks = _empty_levels()
        ob.consume_deltas(empty_asks, tighter_bid)

        # Best bid unchanged since 100.005 rounds to 10000 ticks = 100.00
        best_bid, best_ask = _bbo_prices(ob)
        assert best_bid == pytest.approx(100.00)

        # Add an actually tighter ask at 100.00 (which is lower than 100.01)
        tighter_ask, _ = _make_levels(
            prices=[100.00], sizes=[0.5], norders=[1], with_precision=True
        )
        empty_bids = _empty_bid_levels()
        ob.consume_deltas(tighter_ask, empty_bids)

        best_bid, best_ask = _bbo_prices(ob)
        assert best_ask == pytest.approx(100.00)

        # Delete the second-level bid (99.99)
        del_bids, _ = _make_levels(
            prices=[99.99], sizes=[0.0], norders=[0], with_precision=True
        )
        # Delete the second-level ask (100.02)
        del_asks, _ = _make_levels(
            prices=[100.02], sizes=[0.0], norders=[0], with_precision=True
        )
        ob.consume_deltas(del_asks, del_bids)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        bid_prices = list(bid_arr["price"])
        ask_prices = list(ask_arr["price"])
        assert bid_prices == sorted(bid_prices, reverse=True)
        assert ask_prices == sorted(ask_prices)

    def test_mixed_bbo_and_regular_updates(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.00 - i * 0.01 for i in range(5)],
            sizes=[1.0] * 5,
            norders=[1] * 5,
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.01 + i * 0.01 for i in range(5)],
            sizes=[1.0] * 5,
            norders=[1] * 5,
            with_precision=True,
        )
        ob.consume_snapshot(asks, bids)

        for i in range(10):
            if i % 2 == 0:
                new_bid = 100.00 + (i * 0.001)
                new_ask = 100.01 + (i * 0.001)
                ob.consume_bbo(
                    PyOrderbookLevel.with_ticks_and_lots(
                        new_ask, 1.0, TICK_SIZE, LOT_SIZE, 1
                    ),
                    PyOrderbookLevel.with_ticks_and_lots(
                        new_bid, 1.0, TICK_SIZE, LOT_SIZE, 1
                    ),
                )
            else:
                new_bid = 99.95 - (i * 0.01)
                new_ask = 100.06 + (i * 0.01)
                bid_d, _ = _make_levels([new_bid], [1.0], [1], with_precision=True)
                ask_d, _ = _make_levels([new_ask], [1.0], [1], with_precision=True)
                ob.consume_deltas(ask_d, bid_d)

            bid_arr, ask_arr = _bids_asks_arrays(ob)
            assert bid_arr["price"][0] <= ask_arr["price"][0]
            assert list(bid_arr["price"]) == sorted(bid_arr["price"], reverse=True)
            assert list(ask_arr["price"]) == sorted(ask_arr["price"])

    def test_floating_point_precision(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.01, 100.00, 99.99],
            sizes=[1.0, 1.0, 1.0],
            norders=[1, 1, 1],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.02, 100.03, 100.04],
            sizes=[1.0, 1.0, 1.0],
            norders=[1, 1, 1],
            with_precision=True,
        )
        ob.consume_snapshot(asks, bids)

        spread = ob.get_bbo_spread()
        assert spread == pytest.approx(TICK_SIZE)
        assert abs(spread - 0.01) < 1e-15

    def test_empty_orderbook_errors(self):
        ob = _mk_book(num_levels=64)
        with pytest.raises((RuntimeError, ValueError)):
            ob.get_bbo()
        with pytest.raises((RuntimeError, ValueError)):
            ob.get_bids()
        with pytest.raises((RuntimeError, ValueError)):
            ob.get_asks()
        with pytest.raises((RuntimeError, ValueError)):
            ob.get_bbo_spread()

    def test_reset_functionality(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[100.00, 99.99, 99.98],
            sizes=[1.0, 1.0, 1.0],
            norders=[1, 1, 1],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03],
            sizes=[1.0, 1.0, 1.0],
            norders=[1, 1, 1],
            with_precision=True,
        )
        ob.consume_snapshot(asks, bids)
        ob.clear()

        with pytest.raises((RuntimeError, ValueError)):
            ob.get_mid_price()
        with pytest.raises((RuntimeError, ValueError)):
            ob.get_bbo_spread()


class TestPrecisionConsistency:
    """Test precision consistency of conversions and spreads."""

    def test_tick_precision_consistency(self):
        test_prices = [100.00, 100.01, 100.99, 99.99, 0.01, 999.99]
        for price in test_prices:
            ticks = convert_price_to_tick(price, TICK_SIZE)
            recovered_price = convert_price_from_tick(ticks, TICK_SIZE)
            assert abs(recovered_price - (ticks * TICK_SIZE)) < 1e-15

    def test_lot_precision_consistency(self):
        test_sizes = [1.0, 0.5, 0.001, 1.999, 0.0001]
        for size in test_sizes:
            lots = convert_size_to_lot(size, LOT_SIZE)
            recovered_size = convert_size_from_lot(lots, LOT_SIZE)
            assert abs(recovered_size - (lots * LOT_SIZE)) < 1e-15

    def test_spread_calculation_precision(self):
        ob = _mk_book(num_levels=64)
        bids, _ = _make_levels(
            prices=[99.99, 99.98, 99.97],
            sizes=[1.0, 1.0, 1.0],
            norders=[1, 1, 1],
            with_precision=True,
        )
        asks, _ = _make_levels(
            prices=[100.01, 100.02, 100.03],
            sizes=[1.0, 1.0, 1.0],
            norders=[1, 1, 1],
            with_precision=True,
        )
        ob.consume_snapshot(asks, bids)

        spread = ob.get_bbo_spread()
        best_bid_ticks = convert_price_to_tick(99.99, TICK_SIZE)
        best_ask_ticks = convert_price_to_tick(100.01, TICK_SIZE)
        spread_ticks = best_ask_ticks - best_bid_ticks

        assert spread_ticks == 2
        assert spread == pytest.approx(spread_ticks * TICK_SIZE)


class TestNumpyIngestion:
    """Test direct numpy array ingestion methods."""

    def test_basic_snapshot_with_numpy_arrays(self):
        """Test basic snapshot ingestion using numpy arrays."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == [100.0, 99.99, 99.98]
        assert list(ask_arr["price"]) == [100.01, 100.02, 100.03]
        assert list(bid_arr["size"]) == pytest.approx([1.0, 2.0, 3.0])
        assert list(ask_arr["size"]) == pytest.approx([1.5, 2.5, 3.5])
        assert ob.get_bbo_spread() == pytest.approx(0.01)

    def test_snapshot_with_norders(self):
        """Test snapshot ingestion with explicit norders arrays."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        bid_norders = np.array([5, 10], dtype=np.uint64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)
        ask_norders = np.array([3, 7], dtype=np.uint64)

        ob.consume_snapshot_numpy(
            ask_prices, ask_sizes, bid_prices, bid_sizes, ask_norders, bid_norders
        )

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["norders"]) == [5, 10]
        assert list(ask_arr["norders"]) == [3, 7]

    def test_snapshot_without_norders_defaults_to_one(self):
        """Test that norders defaults to 1 when not provided."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0, 99.99], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert list(bid_arr["norders"]) == [1, 1]
        assert list(ask_arr["norders"]) == [1, 1]

    def test_basic_deltas_with_numpy_arrays(self):
        """Test basic delta ingestion using numpy arrays."""
        ob = _mk_book(num_levels=64)

        # Initial snapshot
        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Apply delta: add a new ask level
        delta_ask_prices = np.array([100.04], dtype=np.float64)
        delta_ask_sizes = np.array([1.0], dtype=np.float64)
        # Use far-away price with zero size for no-op bid delta
        delta_bid_prices = np.array([1.0], dtype=np.float64)
        delta_bid_sizes = np.array([0.0], dtype=np.float64)

        ob.consume_deltas_numpy(
            delta_ask_prices, delta_ask_sizes, delta_bid_prices, delta_bid_sizes
        )

        _, ask_arr = _bids_asks_arrays(ob)
        assert list(ask_arr["price"]) == [100.01, 100.02, 100.03, 100.04]

    def test_deltas_modify_existing_level(self):
        """Test delta modifying an existing level."""
        ob = _mk_book(num_levels=64)

        # Initial snapshot
        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Modify the best bid
        delta_bid_prices = np.array([100.0], dtype=np.float64)
        delta_bid_sizes = np.array([5.0], dtype=np.float64)
        delta_bid_norders = np.array([10], dtype=np.uint64)
        # No-op for asks
        delta_ask_prices = np.array([999.0], dtype=np.float64)
        delta_ask_sizes = np.array([0.0], dtype=np.float64)

        ob.consume_deltas_numpy(
            delta_ask_prices,
            delta_ask_sizes,
            delta_bid_prices,
            delta_bid_sizes,
            bid_norders=delta_bid_norders,
        )

        bid_arr, _ = _bids_asks_arrays(ob)
        assert bid_arr["size"][0] == pytest.approx(5.0)
        assert bid_arr["norders"][0] == 10

    def test_deltas_delete_level_with_zero_size(self):
        """Test delta deleting a level by setting size to zero."""
        ob = _mk_book(num_levels=64)

        # Initial snapshot
        bid_prices = np.array([100.0, 99.99, 99.98], dtype=np.float64)
        bid_sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02, 100.03], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Delete the best bid
        delta_bid_prices = np.array([100.0], dtype=np.float64)
        delta_bid_sizes = np.array([0.0], dtype=np.float64)
        # No-op for asks
        delta_ask_prices = np.array([999.0], dtype=np.float64)
        delta_ask_sizes = np.array([0.0], dtype=np.float64)

        ob.consume_deltas_numpy(
            delta_ask_prices, delta_ask_sizes, delta_bid_prices, delta_bid_sizes
        )

        bid_arr, _ = _bids_asks_arrays(ob)
        assert list(bid_arr["price"]) == [99.99, 99.98]

    def test_numpy_vs_pyorderbooklevels_equivalence(self):
        """Test that numpy ingestion produces identical results to PyOrderbookLevels ingestion."""
        ob_numpy = _mk_book(num_levels=64)
        ob_py = _mk_book(num_levels=64)

        bid_prices_list = [100.0, 99.99, 99.98, 99.97]
        bid_sizes_list = [1.0, 2.0, 3.0, 4.0]
        ask_prices_list = [100.01, 100.02, 100.03, 100.04]
        ask_sizes_list = [1.5, 2.5, 3.5, 4.5]

        # Numpy ingestion
        bid_prices = np.array(bid_prices_list, dtype=np.float64)
        bid_sizes = np.array(bid_sizes_list, dtype=np.float64)
        ask_prices = np.array(ask_prices_list, dtype=np.float64)
        ask_sizes = np.array(ask_sizes_list, dtype=np.float64)
        ob_numpy.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # PyOrderbookLevels ingestion
        bids_py, _ = _make_levels(
            prices=bid_prices_list,
            sizes=bid_sizes_list,
            norders=[1] * 4,
            with_precision=False,
        )
        asks_py, _ = _make_levels(
            prices=ask_prices_list,
            sizes=ask_sizes_list,
            norders=[1] * 4,
            with_precision=False,
        )
        ob_py.consume_snapshot(asks_py, bids_py)

        # Compare results
        bid_arr_numpy, ask_arr_numpy = _bids_asks_arrays(ob_numpy)
        bid_arr_py, ask_arr_py = _bids_asks_arrays(ob_py)

        assert list(bid_arr_numpy["price"]) == list(bid_arr_py["price"])
        assert list(ask_arr_numpy["price"]) == list(ask_arr_py["price"])
        assert list(bid_arr_numpy["size"]) == pytest.approx(list(bid_arr_py["size"]))
        assert list(ask_arr_numpy["size"]) == pytest.approx(list(ask_arr_py["size"]))
        assert ob_numpy.get_mid_price() == ob_py.get_mid_price()
        assert ob_numpy.get_bbo_spread() == ob_py.get_bbo_spread()

    def test_empty_arrays_snapshot(self):
        """Test snapshot with empty arrays for one side."""
        ob = _mk_book(num_levels=64)

        # Only populate asks, empty bids
        bid_prices = np.array([], dtype=np.float64)
        bid_sizes = np.array([], dtype=np.float64)
        ask_prices = np.array([100.01, 100.02], dtype=np.float64)
        ask_sizes = np.array([1.5, 2.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        # Should have asks but no bids
        _, ask_arr = _bids_asks_arrays(ob)
        assert len(ask_arr) == 2

        # Orderbook is half-empty, so mid_price should raise
        with pytest.raises(RuntimeError):
            ob.get_mid_price()

    def test_single_level_numpy(self):
        """Test numpy ingestion with single level per side."""
        ob = _mk_book(num_levels=64)

        bid_prices = np.array([100.0], dtype=np.float64)
        bid_sizes = np.array([1.0], dtype=np.float64)
        ask_prices = np.array([100.01], dtype=np.float64)
        ask_sizes = np.array([1.5], dtype=np.float64)

        ob.consume_snapshot_numpy(ask_prices, ask_sizes, bid_prices, bid_sizes)

        bid_arr, ask_arr = _bids_asks_arrays(ob)
        assert len(bid_arr) == 1
        assert len(ask_arr) == 1
        assert bid_arr["price"][0] == 100.0
        assert ask_arr["price"][0] == 100.01

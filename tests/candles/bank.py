"""Deterministic, independently specified candle aggregation scenarios.

Expected groups in this module describe candle membership directly. They are
not produced by running a candle implementation, so the same data can exercise
streaming and historical replay paths without using production output as its
own oracle.
"""

from dataclasses import dataclass
import math
from typing import TypeVar

from mm_toolbox.candles.base import BaseCandles, Candle, Trade


LimitT = TypeVar("LimitT")


def trade(
    timestamp: int,
    size: float = 1.0,
    price: float = 100.0,
    is_buy: bool = True,
) -> Trade:
    """Build a trade for deterministic event-time scenarios."""
    return Trade(time_ms=timestamp, is_buy=is_buy, price=price, size=size)


@dataclass(frozen=True)
class Scenario[LimitT]:
    """Input trades and literal expected closed/current trade memberships."""

    name: str
    limit: LimitT
    trades: tuple[Trade, ...]
    closed: tuple[tuple[Trade, ...], ...]
    current: tuple[Trade, ...]


@dataclass(frozen=True)
class MultiLimits:
    """Limits passed to ``MultiCandles``."""

    seconds: float
    ticks: int
    size: float


def _scenario(
    name: str,
    limit: LimitT,
    trades: tuple[Trade, ...],
    closed: tuple[tuple[Trade, ...], ...] = (),
    current: tuple[Trade, ...] = (),
) -> Scenario[LimitT]:
    return Scenario(name, limit, trades, closed, current)


def _chunks(items: tuple[Trade, ...], size: int) -> tuple[tuple[Trade, ...], ...]:
    """Partition a known deterministic pattern into fixed-size groups."""
    return tuple(items[index : index + size] for index in range(0, len(items), size))


def _piece(source: Trade, size: float) -> Trade:
    """Specify the expected split value for part of a source trade."""
    return trade(source.time_ms, size, source.price, source.is_buy)


def _assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0), (
        f"{actual!r} != {expected!r}"
    )


def assert_candle(
    candle: Candle,
    expected: tuple[Trade, ...],
    store_trades: bool = True,
) -> None:
    """Check all candle summaries against independent expected membership."""
    if not expected:
        assert candle == Candle.empty()
        return

    sizes = tuple(item.size for item in expected)
    values = tuple(item.price * item.size for item in expected)
    buy_sizes = tuple(item.size for item in expected if item.is_buy)
    sell_sizes = tuple(item.size for item in expected if not item.is_buy)
    buy_values = tuple(item.price * item.size for item in expected if item.is_buy)
    sell_values = tuple(item.price * item.size for item in expected if not item.is_buy)
    total_size = math.fsum(sizes)
    total_value = math.fsum(values)

    assert candle.open_time_ms == expected[0].time_ms
    assert candle.close_time_ms == expected[-1].time_ms
    assert candle.open_price == expected[0].price
    assert candle.high_price == max(item.price for item in expected)
    assert candle.low_price == min(item.price for item in expected)
    assert candle.close_price == expected[-1].price
    _assert_close(candle.buy_size, math.fsum(buy_sizes))
    _assert_close(candle.buy_volume, math.fsum(buy_values))
    _assert_close(candle.sell_size, math.fsum(sell_sizes))
    _assert_close(candle.sell_volume, math.fsum(sell_values))
    _assert_close(candle.vwap, total_value / total_size if total_size else 0.0)
    assert candle.num_trades == len(expected)
    if store_trades:
        assert len(candle.trades) == len(expected)
        for actual_trade, expected_trade in zip(candle.trades, expected, strict=True):
            assert actual_trade.time_ms == expected_trade.time_ms
            assert actual_trade.is_buy is expected_trade.is_buy
            assert actual_trade.price == expected_trade.price
            _assert_close(actual_trade.size, expected_trade.size)
    else:
        assert candle.trades == []


def assert_scenario(
    candles: BaseCandles,
    scenario: Scenario[LimitT],
    store_trades: bool = True,
) -> None:
    """Compare the full closed history and the in-progress candle."""
    assert len(candles) == len(scenario.closed), scenario.name
    for index, expected in enumerate(scenario.closed):
        assert_candle(candles[index], expected, store_trades)
    assert_candle(candles.latest_candle, scenario.current, store_trades)


# Tick count scenarios. Long patterns use fixed slicing, which is the
# mathematical definition of these cases rather than a second aggregator.
TICK_SCENARIOS: tuple[Scenario[int], ...] = (
    _scenario("empty", 3, (), current=()),
    _scenario("singleton_open", 3, (trade(10),), current=(trade(10),)),
    _scenario(
        "limit_one_closes_each_trade",
        1,
        (t0 := trade(10, 1, 2), t1 := trade(10, 2, 3, False), t2 := trade(11, 0, 4)),
        ((t0,), (t1,), (t2,)),
    ),
    _scenario(
        "two_exact_and_partial_tail",
        2,
        (t0 := trade(0, 1, 10), t1 := trade(1, 1, 11), t2 := trade(2, 1, 12)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "one_below_threshold",
        4,
        (t0 := trade(1), t1 := trade(2), t2 := trade(3)),
        current=(t0, t1, t2),
    ),
    _scenario(
        "one_above_with_remainder",
        3,
        (t0 := trade(1), t1 := trade(2), t2 := trade(3), t3 := trade(4)),
        ((t0, t1, t2),),
        (t3,),
    ),
    _scenario(
        "multiple_closures_and_tail",
        3,
        (
            t0 := trade(1),
            t1 := trade(2),
            t2 := trade(3),
            t3 := trade(4),
            t4 := trade(5),
            t5 := trade(6),
            t6 := trade(7),
            t7 := trade(8),
        ),
        ((t0, t1, t2), (t3, t4, t5)),
        (t6, t7),
    ),
    _scenario(
        "limit_larger_than_input",
        9,
        (t0 := trade(5), t1 := trade(6), t2 := trade(7)),
        current=(t0, t1, t2),
    ),
    _scenario(
        "all_buys",
        4,
        (
            t0 := trade(10, 0.5, 5),
            t1 := trade(11, 1.5, 7),
            t2 := trade(12, 2.0, 9),
            t3 := trade(13, 0.25, 11),
        ),
        ((t0, t1, t2, t3),),
    ),
    _scenario(
        "all_sells",
        3,
        (
            t0 := trade(10, 2, 8, False),
            t1 := trade(11, 3, 6, False),
            t2 := trade(12, 1, 4, False),
        ),
        ((t0, t1, t2),),
    ),
    _scenario(
        "alternating_sides_unequal_sizes",
        5,
        (
            t0 := trade(3, 0.25, 2, True),
            t1 := trade(3, 1.5, 4, False),
            t2 := trade(4, 2.25, 8, True),
            t3 := trade(7, 0.5, 16, False),
            t4 := trade(8, 3.0, 32, True),
        ),
        ((t0, t1, t2, t3, t4),),
    ),
    _scenario(
        "zero_size_still_counts",
        2,
        (t0 := trade(0, 0, 5), t1 := trade(1, 0, 6, False), t2 := trade(2, 0.5, 7)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "identical_timestamp_boundary",
        2,
        (t0 := trade(9, 1, 1), t1 := trade(9, 1, 2), t2 := trade(9, 1, 3)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "negative_time_and_epoch_zero",
        3,
        (t0 := trade(-50, 1, 0), t1 := trade(-50, 1, -2), t2 := trade(0, 1, 2)),
        ((t0, t1, t2),),
    ),
    _scenario(
        "stale_before_and_after_close_ignored",
        2,
        (
            t0 := trade(10, 1, 10),
            t1 := trade(11, 1, 11),
            trade(9, 999, 9999),
            t2 := trade(12, 1, 12),
            trade(11, 999, -9999),
            t3 := trade(13, 1, 13),
        ),
        ((t0, t1), (t2, t3)),
    ),
    _scenario(
        "reused_trade_object_counts_twice",
        2,
        (shared := trade(5, 0.75, 7), shared, t2 := trade(6, 1.25, 9)),
        ((shared, shared),),
        (t2,),
    ),
    _scenario(
        "zero_and_negative_prices",
        3,
        (t0 := trade(1, 1, 0), t1 := trade(2, 2, -3, False), t2 := trade(3, 0.5, 4)),
        ((t0, t1, t2),),
    ),
    _scenario(
        "large_prices_tiny_sizes",
        2,
        (
            t0 := trade(1, 1e-12, 1e150),
            t1 := trade(2, 2e-12, 5e149),
            t2 := trade(3, 1e-15, 1e150),
        ),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "upward_prices",
        4,
        (
            t0 := trade(0, 1, 1),
            t1 := trade(1, 2, 2),
            t2 := trade(2, 3, 3),
            t3 := trade(3, 4, 4),
        ),
        ((t0, t1, t2, t3),),
    ),
    _scenario(
        "downward_prices",
        4,
        (
            t0 := trade(0, 1, 4),
            t1 := trade(1, 2, 3),
            t2 := trade(2, 3, 2),
            t3 := trade(3, 4, 1),
        ),
        ((t0, t1, t2, t3),),
    ),
    _scenario(
        "zigzag_prices_and_side_changes",
        6,
        (
            t0 := trade(0, 0.1, 10),
            t1 := trade(1, 0.2, 20, False),
            t2 := trade(2, 0.3, 5),
            t3 := trade(3, 0.4, 15, False),
            t4 := trade(4, 0.5, 7),
            t5 := trade(5, 0.6, 12, False),
        ),
        ((t0, t1, t2, t3, t4, t5),),
    ),
    _scenario(
        "repeated_closures_reset_summary",
        2,
        (
            t0 := trade(1, 2, 100),
            t1 := trade(2, 3, 101, False),
            t2 := trade(3, 4, -5),
            t3 := trade(4, 5, -6, False),
            t4 := trade(5, 6, 0),
        ),
        ((t0, t1), (t2, t3)),
        (t4,),
    ),
    _scenario(
        "medium_35_mixed_trades_groups_of_four",
        4,
        _medium_tick_35 := tuple(
            trade(index, 0.25 + (index % 4) * 0.25, 100 + (index % 9), index % 2 == 0)
            for index in range(35)
        ),
        _chunks(_medium_tick_35, 4)[:-1],
        _chunks(_medium_tick_35, 4)[-1],
    ),
    _scenario(
        "medium_24_unequal_sides_groups_of_six",
        6,
        _medium_tick_24 := tuple(
            trade(index, 0.125 + (index % 5) * 0.375, (index % 13) - 6, index % 3 != 0)
            for index in range(24)
        ),
        _chunks(_medium_tick_24, 6),
    ),
    _scenario(
        "medium_20_zero_size_events_count_as_ticks",
        5,
        _medium_tick_zero_20 := tuple(
            trade(index, 0 if index % 3 == 0 else 0.25, 1 + index, index % 2 == 0)
            for index in range(20)
        ),
        _chunks(_medium_tick_zero_20, 5),
    ),
    _scenario(
        "medium_18_equal_timestamp_blocks",
        7,
        _medium_tick_equal_18 := tuple(
            trade(index // 6, 0.5 + (index % 3) * 0.25, 10 + index, index % 2 == 0)
            for index in range(18)
        ),
        _chunks(_medium_tick_equal_18, 7)[:-1],
        _chunks(_medium_tick_equal_18, 7)[-1],
    ),
    _scenario(
        "long_257_events_groups_of_seven",
        7,
        _long_tick_257 := tuple(
            trade(
                index, 0.125 + (index % 5) * 0.25, 10 + (index % 11) - 5, index % 2 == 0
            )
            for index in range(257)
        ),
        _chunks(_long_tick_257, 7)[:-1],
        _chunks(_long_tick_257, 7)[-1],
    ),
    _scenario(
        "long_1024_events_groups_of_thirteen",
        13,
        _long_tick_1024 := tuple(
            trade(
                index,
                0 if index % 17 == 0 else 0.5,
                100 + (index % 31) / 8,
                index % 3 != 0,
            )
            for index in range(1024)
        ),
        _chunks(_long_tick_1024, 13)[:-1],
        _chunks(_long_tick_1024, 13)[-1],
    ),
    _scenario(
        "long_4097_single_tick_candles",
        1,
        _long_tick_4097 := tuple(
            trade(index, (index % 7) / 8, (index % 19) - 9, index % 2 == 0)
            for index in range(4097)
        ),
        tuple((item,) for item in _long_tick_4097),
    ),
)


# Time scenarios use schedules anchored at the first accepted trade. The
# expected slices below follow fixed timestamp boundaries, including empty
# intervals being skipped instead of materialized.
TIME_SCENARIOS: tuple[Scenario[float], ...] = (
    _scenario("empty", 1.0, ()),
    _scenario("singleton_open", 1.0, (t0 := trade(17),), current=(t0,)),
    _scenario(
        "just_below_at_and_after_boundary",
        1.0,
        (
            t0 := trade(0, 1, 10),
            t1 := trade(999, 2, 11),
            t2 := trade(1000, 3, 12),
            t3 := trade(1001, 4, 13),
            t4 := trade(2000, 5, 14),
        ),
        ((t0, t1), (t2, t3)),
        (t4,),
    ),
    _scenario(
        "boundary_trade_starts_next_candle",
        0.005,
        (
            t0 := trade(100, 1, 1),
            t1 := trade(104, 1, 2),
            t2 := trade(105, 1, 3),
            t3 := trade(106, 1, 4),
        ),
        ((t0, t1),),
        (t2, t3),
    ),
    _scenario(
        "several_consecutive_boundaries",
        0.001,
        (
            t0 := trade(1, 1, 10),
            t1 := trade(2, 2, 11),
            t2 := trade(3, 3, 12),
            t3 := trade(4, 4, 13),
            t4 := trade(5, 5, 14),
        ),
        ((t0,), (t1,), (t2,), (t3,)),
        (t4,),
    ),
    _scenario(
        "sparse_gap_keeps_original_schedule",
        1.0,
        (
            t0 := trade(0, 1, 10),
            t1 := trade(3000, 2, 20),
            t2 := trade(3999, 3, 30),
            t3 := trade(4000, 4, 40),
        ),
        ((t0,), (t1, t2)),
        (t3,),
    ),
    _scenario(
        "half_millisecond_duration",
        0.0005,
        (
            t0 := trade(0, 1, 1),
            t1 := trade(1, 2, 2),
            t2 := trade(2, 3, 3),
            t3 := trade(3, 4, 4),
        ),
        ((t0,), (t1,), (t2,)),
        (t3,),
    ),
    _scenario(
        "one_millisecond_duration",
        0.001,
        (
            t0 := trade(50, 1, 1),
            t1 := trade(50, 1, 2),
            t2 := trade(51, 1, 3),
            t3 := trade(52, 1, 4),
        ),
        ((t0, t1), (t2,)),
        (t3,),
    ),
    _scenario(
        "huge_idle_gap_skips_empty_buckets",
        2.0,
        (
            t0 := trade(100, 1, 10),
            t1 := trade(1000000, 2, 20),
            t2 := trade(1000099, 3, 30),
            t3 := trade(1000100, 4, 40),
        ),
        ((t0,), (t1, t2)),
        (t3,),
    ),
    _scenario(
        "identical_timestamps_share_bucket",
        0.01,
        (
            t0 := trade(5, 1, 10),
            t1 := trade(5, 2, 9, False),
            t2 := trade(5, 3, 11),
            t3 := trade(5, 4, 8, False),
        ),
        current=(t0, t1, t2, t3),
    ),
    _scenario(
        "epoch_zero_anchor",
        2.0,
        (t0 := trade(0, 1, 0), t1 := trade(1999, 2, -2), t2 := trade(2000, 3, 4)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "negative_anchor_away_from_zero_sentinel",
        1.0,
        (
            t0 := trade(-5000, 1, -10),
            t1 := trade(-4001, 2, -9),
            t2 := trade(-4000, 3, -8),
        ),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "zero_size_boundary_trade_is_retained",
        1.0,
        (
            t0 := trade(10, 1, 1),
            t1 := trade(1009, 0, 2, False),
            t2 := trade(1010, 2, 3),
        ),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "stale_extreme_prices_do_not_change_summary",
        1.0,
        (
            t0 := trade(100, 1, 10),
            t1 := trade(200, 2, 20),
            trade(150, 1e100, 1e100),
            t2 := trade(1100, 3, 30),
            trade(1099, 1e100, -1e100),
            t3 := trade(2100, 4, 40),
        ),
        ((t0, t1), (t2,)),
        (t3,),
    ),
    _scenario(
        "zero_and_negative_prices",
        0.001,
        (t0 := trade(0, 1, 0), t1 := trade(1, 2, -3, False), t2 := trade(2, 3, 4)),
        ((t0,), (t1,)),
        (t2,),
    ),
    _scenario(
        "alternating_sides_and_unequal_sizes",
        10.0,
        (
            t0 := trade(0, 0.25, 2),
            t1 := trade(1000, 1.5, 4, False),
            t2 := trade(2000, 2.25, 8),
            t3 := trade(3000, 0.5, 16, False),
        ),
        current=(t0, t1, t2, t3),
    ),
    _scenario(
        "all_sells_and_flat_prices",
        5.0,
        (
            t0 := trade(1, 1, 7, False),
            t1 := trade(2, 2, 7, False),
            t2 := trade(3, 0.5, 7, False),
        ),
        current=(t0, t1, t2),
    ),
    _scenario(
        "repeated_closures_reset_ohlc_and_vwap",
        1.0,
        (
            t0 := trade(0, 1, 100),
            t1 := trade(1000, 2, 110, False),
            t2 := trade(2000, 3, -10),
            t3 := trade(3000, 4, -20, False),
            t4 := trade(4000, 0.5, 0),
        ),
        ((t0,), (t1,), (t2,), (t3,)),
        (t4,),
    ),
    _scenario(
        "reused_trade_object_with_equal_timestamp",
        0.002,
        (shared := trade(50, 1.5, 10), shared, t2 := trade(52, 0.25, 20)),
        ((shared, shared),),
        (t2,),
    ),
    _scenario(
        "small_duration_at_large_timestamp",
        0.0005,
        (t0 := trade(10**12, 1, 1e12), t1 := trade(10**12 + 1, 2, 1e12 + 1)),
        ((t0,),),
        (t1,),
    ),
    _scenario(
        "negative_start_boundary_exactly_zero",
        1.0,
        (t0 := trade(-1000, 1, 10), t1 := trade(0, 2, 11), t2 := trade(1, 3, 12)),
        ((t0,),),
        (t1, t2),
    ),
    _scenario(
        "medium_40_events_two_second_windows",
        2.0,
        _medium_time_40 := tuple(
            trade(
                index * 250, 0.25 + (index % 4) * 0.25, (index % 15) - 7, index % 2 == 0
            )
            for index in range(40)
        ),
        _chunks(_medium_time_40, 8)[:-1],
        _chunks(_medium_time_40, 8)[-1],
    ),
    _scenario(
        "medium_30_irregular_arrivals_follow_anchor_schedule",
        2.5,
        _medium_time_30 := tuple(
            trade(index * 175, 0.5 + (index % 3) * 0.25, 10 + index, index % 2 == 0)
            for index in range(30)
        ),
        (_medium_time_30[:15], _medium_time_30[15:29]),
        (_medium_time_30[29],),
    ),
    _scenario(
        "medium_24_equal_timestamp_boundary_blocks",
        0.01,
        _medium_time_equal_24 := tuple(
            trade(
                (index // 6) * 10, 0.5 + (index % 4) * 0.25, 10 + index, index % 2 == 0
            )
            for index in range(24)
        ),
        _chunks(_medium_time_equal_24, 6)[:-1],
        _chunks(_medium_time_equal_24, 6)[-1],
    ),
    _scenario(
        "medium_18_sparse_arrivals_close_two_event_windows",
        3.0,
        _medium_time_sparse_18 := tuple(
            trade(index * 1500, 0.5 + (index % 3) * 0.25, 5 + index, index % 2 == 0)
            for index in range(18)
        ),
        _chunks(_medium_time_sparse_18, 2)[:-1],
        _chunks(_medium_time_sparse_18, 2)[-1],
    ),
    _scenario(
        "long_257_events_25ms_schedule",
        0.025,
        _long_time_257 := tuple(
            trade(index, 0.25 + (index % 7) * 0.125, (index % 17) - 8, index % 2 == 0)
            for index in range(257)
        ),
        _chunks(_long_time_257, 25)[:-1],
        _chunks(_long_time_257, 25)[-1],
    ),
    _scenario(
        "long_1024_events_64ms_schedule",
        0.064,
        _long_time_1024 := tuple(
            trade(
                index,
                0 if index % 23 == 0 else 0.5,
                50 + (index % 29) / 4,
                index % 3 != 0,
            )
            for index in range(1024)
        ),
        _chunks(_long_time_1024, 64)[:-1],
        _chunks(_long_time_1024, 64)[-1],
    ),
)


# Price scenarios use binary-exact thresholds and prices whenever a boundary
# distinction matters. The crossing trade belongs to the candle it closes.
PRICE_SCENARIOS: tuple[Scenario[float], ...] = (
    _scenario("empty", 1.0, ()),
    _scenario("singleton_open", 1.0, (t0 := trade(1, 1, 4),), current=(t0,)),
    _scenario(
        "strict_bounds_then_up_crossing_trade_closes",
        0.5,
        (
            t0 := trade(0, 1, 4),
            t1 := trade(1, 1, 4.5),
            t2 := trade(2, 1, 3.5),
            t3 := trade(3, 1, 4.75),
        ),
        ((t0, t1, t2, t3),),
    ),
    _scenario(
        "down_crossing_closes_current_candle",
        0.5,
        (t0 := trade(0, 1, 4), t1 := trade(1, 1, 3.5), t2 := trade(2, 1, 3.25)),
        ((t0, t1, t2),),
    ),
    _scenario(
        "both_exact_bounds_do_not_close",
        0.5,
        (t0 := trade(0, 1, 4), t1 := trade(1, 2, 4.5), t2 := trade(2, 3, 3.5)),
        current=(t0, t1, t2),
    ),
    _scenario(
        "oscillation_inside_bounds",
        1.0,
        (
            t0 := trade(0, 1, 10),
            t1 := trade(1, 2, 10.5),
            t2 := trade(2, 3, 9.25),
            t3 := trade(3, 4, 10.75),
            t4 := trade(4, 5, 9),
        ),
        current=(t0, t1, t2, t3, t4),
    ),
    _scenario(
        "overshoot_closes_once_without_synthetic_bars",
        1.0,
        (t0 := trade(0, 1, 2), t1 := trade(1, 1, 100), t2 := trade(2, 1, 100.5)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "bounds_reset_from_crossing_then_next_opening_trade",
        1.0,
        (
            t0 := trade(0, 1, 2),
            t1 := trade(1, 1, 3.25),
            t2 := trade(2, 1, 4),
            t3 := trade(3, 1, 4.5),
        ),
        ((t0, t1),),
        (t2, t3),
    ),
    _scenario(
        "negative_and_zero_opening_prices",
        0.5,
        (
            t0 := trade(0, 1, -1),
            t1 := trade(1, 1, -0.5),
            t2 := trade(2, 1, -1.5),
            t3 := trade(3, 1, 0),
        ),
        ((t0, t1, t2, t3),),
    ),
    _scenario(
        "zero_size_trade_can_trigger",
        0.5,
        (t0 := trade(0, 1, 2), t1 := trade(1, 0, 3, False), t2 := trade(2, 1, 3)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "size_and_time_gaps_do_not_affect_trigger",
        0.5,
        (
            t0 := trade(0, 1e-12, 1, True),
            t1 := trade(100000, 1e12, 1.25, False),
            t2 := trade(200000, 1e-9, 1.75),
        ),
        ((t0, t1, t2),),
    ),
    _scenario(
        "stale_extreme_trigger_is_ignored",
        0.5,
        (
            t0 := trade(10, 1, 4),
            t1 := trade(11, 1, 4.25),
            trade(9, 1e100, 1e100),
            t2 := trade(12, 1, 4.75),
        ),
        ((t0, t1, t2),),
    ),
    _scenario(
        "equal_timestamp_crossing_after_boundary",
        0.25,
        (t0 := trade(5, 1, 2), t1 := trade(5, 2, 2.25), t2 := trade(5, 3, 2.5)),
        ((t0, t1, t2),),
    ),
    _scenario(
        "repeated_bidirectional_closures_reset_bounds",
        0.5,
        (
            t0 := trade(0, 1, 4),
            t1 := trade(1, 1, 4.75),
            t2 := trade(2, 1, 4.5),
            t3 := trade(3, 1, 3.75),
            t4 := trade(4, 1, 4),
        ),
        ((t0, t1), (t2, t3)),
        (t4,),
    ),
    _scenario(
        "tiny_binary_threshold",
        0.125,
        (
            t0 := trade(0, 1, 0),
            t1 := trade(1, 2, 0.0625),
            t2 := trade(2, 3, 0.125),
            t3 := trade(3, 4, 0.1875),
        ),
        ((t0, t1, t2, t3),),
    ),
    _scenario(
        "large_prices_and_small_sizes",
        2.0,
        (
            t0 := trade(0, 1e-100, 2**52),
            t1 := trade(1, 2e-100, 2**52 + 2),
            t2 := trade(2, 1e-100, 2**52 + 4),
        ),
        ((t0, t1, t2),),
    ),
    _scenario(
        "all_sells_with_nontrivial_vwap",
        1.0,
        (
            t0 := trade(0, 0.25, 8, False),
            t1 := trade(1, 1.5, 8.5, False),
            t2 := trade(2, 0.75, 9, False),
        ),
        current=(t0, t1, t2),
    ),
    _scenario(
        "alternating_sides_and_unequal_sizes",
        1.0,
        (
            t0 := trade(0, 0.25, 2),
            t1 := trade(1, 2, 2.25, False),
            t2 := trade(2, 0.5, 1.75),
            t3 := trade(3, 3, 2.5, False),
        ),
        current=(t0, t1, t2, t3),
    ),
    _scenario(
        "flat_price_stream",
        0.5,
        (t0 := trade(0, 1, 7), t1 := trade(1, 2, 7), t2 := trade(2, 3, 7)),
        current=(t0, t1, t2),
    ),
    _scenario(
        "reused_trade_object",
        2.0,
        (shared := trade(7, 0.5, 2), shared, t2 := trade(8, 0.25, 3.5)),
        current=(shared, shared, t2),
    ),
    _scenario(
        "signed_price_values_cancel_to_zero_vwap",
        1.0,
        (t0 := trade(0, 1, 4), t1 := trade(1, 1, -4, False)),
        ((t0, t1),),
    ),
    _scenario(
        "medium_40_rising_events_close_every_six",
        1.0,
        _medium_price_40 := tuple(
            trade(index, 0.25 + (index % 3) * 0.125, 10 + index * 0.25, index % 2 == 0)
            for index in range(40)
        ),
        _chunks(_medium_price_40, 6)[:-1],
        _chunks(_medium_price_40, 6)[-1],
    ),
    _scenario(
        "medium_24_repeating_bidirectional_crossings",
        0.5,
        _medium_price_24 := tuple(
            trade(
                index,
                0.25 + (index % 3) * 0.125,
                (0.0, 0.5, -0.5, 0.75)[index % 4],
                index % 2 == 0,
            )
            for index in range(24)
        ),
        _chunks(_medium_price_24, 4),
    ),
    _scenario(
        "medium_25_exact_bounds_never_cross",
        0.5,
        _medium_price_bounds_25 := tuple(
            trade(
                index,
                0.5 + (index % 3) * 0.25,
                0 if index == 0 else (0.5 if index % 2 else -0.5),
                index % 2 == 0,
            )
            for index in range(25)
        ),
        current=_medium_price_bounds_25,
    ),
    _scenario(
        "long_257_rising_events_close_every_four",
        1.0,
        _long_price_257 := tuple(
            trade(index, 0.25 + (index % 3) * 0.125, 10 + index * 0.5, index % 2 == 0)
            for index in range(257)
        ),
        _chunks(_long_price_257, 4)[:-1],
        _chunks(_long_price_257, 4)[-1],
    ),
    _scenario(
        "long_1024_falling_events_close_every_four",
        1.0,
        _long_price_1024 := tuple(
            trade(index, 0.5, 1000 - index * 0.5, index % 2 != 0)
            for index in range(1024)
        ),
        _chunks(_long_price_1024, 4),
    ),
    _scenario(
        "long_257_alternating_crossing_cycles",
        0.5,
        _long_price_oscillating := tuple(
            trade(
                index,
                0.125 + (index % 5) * 0.125,
                20 + (index % 2) * 0.75,
                index % 2 == 0,
            )
            for index in range(257)
        ),
        tuple(_long_price_oscillating[index : index + 2] for index in range(0, 256, 2)),
        (_long_price_oscillating[256],),
    ),
)


# Volume expectations include split Trade pieces. A piece is a new Trade value
# when the source trade straddles a bucket boundary; complete unsplit trades
# remain the source object in expected membership.
VOLUME_SCENARIOS: tuple[Scenario[float], ...] = (
    _scenario("empty", 1.0, ()),
    _scenario("singleton_partial", 2.0, (t0 := trade(1, 1, 5),), current=(t0,)),
    _scenario(
        "exact_threshold",
        2.0,
        (t0 := trade(0, 0.75, 2), t1 := trade(1, 1.25, 4)),
        ((t0, t1),),
    ),
    _scenario(
        "just_below_threshold",
        2.0,
        (t0 := trade(0, 0.5, 2), t1 := trade(1, 1.25, 4)),
        current=(t0, t1),
    ),
    _scenario(
        "one_over_threshold_splits_with_remainder",
        2.0,
        (t0 := trade(0, 2.5, 3),),
        ((p0 := trade(0, 2, 3),),),
        (p1 := trade(0, 0.5, 3),),
    ),
    _scenario(
        "one_trade_spans_many_full_buckets",
        2.0,
        (t0 := trade(10, 8, 5, False),),
        tuple((trade(10, 2, 5, False),) for _ in range(4)),
    ),
    _scenario(
        "partially_occupied_then_oversized_trade",
        3.0,
        (t0 := trade(0, 1, 10), t1 := trade(1, 5, 20, False)),
        ((t0, p0 := trade(1, 2, 20, False)), (p1 := trade(1, 3, 20, False),)),
    ),
    _scenario(
        "mixed_sides_split_across_boundaries",
        2.0,
        (t0 := trade(0, 1.25, 4), t1 := trade(1, 1.5, 8, False)),
        ((t0, p0 := trade(1, 0.75, 8, False)),),
        (p1 := trade(1, 0.75, 8, False),),
    ),
    _scenario(
        "same_timestamp_split_pieces",
        1.0,
        (t0 := trade(5, 2.5, 4),),
        ((p0 := trade(5, 1, 4),), (p1 := trade(5, 1, 4),)),
        (p2 := trade(5, 0.5, 4),),
    ),
    _scenario(
        "zero_size_events_are_not_members",
        2.0,
        (t0 := trade(0, 1, 5), trade(1, 0, 1000, False), t1 := trade(2, 1, 7, False)),
        ((t0, t1),),
    ),
    _scenario(
        "all_zero_size_stream_is_empty",
        1.0,
        (trade(0, 0, 10), trade(1, 0, -10, False)),
    ),
    _scenario(
        "zero_size_event_advances_stale_clock",
        1.0,
        (t0 := trade(10, 0.5, 2), trade(20, 0, 999), trade(15, 0.5, 4)),
        current=(t0,),
    ),
    _scenario(
        "binary_fractional_sums",
        1.0,
        (
            t0 := trade(0, 0.25, 1),
            t1 := trade(1, 0.25, 2),
            t2 := trade(2, 0.5, 4),
            t3 := trade(3, 0.5, 8),
        ),
        ((t0, t1, t2),),
        (t3,),
    ),
    _scenario(
        "tiny_threshold_split",
        1e-13,
        (t0 := trade(0, 2.5e-13, 10),),
        ((p0 := trade(0, 1e-13, 10),), (p1 := trade(0, 1e-13, 10),)),
        (p2 := trade(0, 0.5e-13, 10),),
    ),
    _scenario(
        "nextafter_below_threshold_closes",
        1.0,
        (t0 := trade(0, math.nextafter(1.0, 0.0), 4),),
        ((t0,),),
    ),
    _scenario(
        "nextafter_limit_then_remainder",
        1.0,
        (t0 := trade(0, math.nextafter(1.0, 0.0) + 0.25, 4),),
        ((p0 := trade(0, 1.0, 4),),),
        (p1 := trade(0, math.nextafter(1.0, 0.0) + 0.25 - 1.0, 4),),
    ),
    _scenario(
        "prices_do_not_affect_size_threshold",
        2.0,
        (t0 := trade(0, 1, 0), t1 := trade(1, 1, 1e100)),
        ((t0, t1),),
    ),
    _scenario(
        "sell_only_threshold",
        1.5,
        (t0 := trade(0, 0.5, 10, False), t1 := trade(1, 1, 6, False)),
        ((t0, t1),),
    ),
    _scenario(
        "stale_trade_does_not_contribute_size",
        2.0,
        (t0 := trade(10, 1, 1), trade(9, 100, 1000), t1 := trade(11, 0.5, 2)),
        current=(t0, t1),
    ),
    _scenario(
        "repeated_closures_reset_side_totals",
        1.0,
        (
            t0 := trade(0, 0.5, 10),
            t1 := trade(1, 0.5, 20, False),
            t2 := trade(2, 1, -5, False),
            t3 := trade(3, 0.25, 8),
        ),
        ((t0, t1), (t2,)),
        (t3,),
    ),
    _scenario(
        "medium_31_quarter_size_events_with_tail",
        1.0,
        _medium_volume_31 := tuple(
            trade(index, 0.25, (index % 13) - 6, index % 2 == 0) for index in range(31)
        ),
        _chunks(_medium_volume_31, 4)[:-1],
        _chunks(_medium_volume_31, 4)[-1],
    ),
    _scenario(
        "long_12_oversized_trades_make_three_split_candles_each",
        1.0,
        _long_volume_bursts_12 := tuple(
            trade(index * 100, 3.0, (index % 7) - 3, index % 2 == 0)
            for index in range(12)
        ),
        tuple(
            (_piece(source, 1.0),)
            for source in _long_volume_bursts_12
            for _ in range(3)
        ),
    ),
    _scenario(
        "decimal_threshold_residual_is_preserved",
        0.1,
        (t0 := trade(0, 0.25, 5, False),),
        ((p0 := trade(0, 0.1, 5, False),), (p1 := trade(0, 0.1, 5, False),)),
        (p2 := trade(0, 0.05, 5, False),),
    ),
    _scenario(
        "all_sells_split_after_existing_partial_size",
        2.0,
        (t0 := trade(0, 1.5, 8, False), t1 := trade(1, 2.25, 4, False)),
        ((t0, p0 := trade(1, 0.5, 4, False)),),
        (p1 := trade(1, 1.75, 4, False),),
    ),
    _scenario(
        "medium_34_binary_eighth_size_events",
        0.5,
        _medium_volume_eighths_34 := tuple(
            trade(index, 0.125, (index % 11) / 2, index % 3 != 0) for index in range(34)
        ),
        _chunks(_medium_volume_eighths_34, 4)[:-1],
        _chunks(_medium_volume_eighths_34, 4)[-1],
    ),
    _scenario(
        "long_1024_quarter_size_events",
        1.0,
        _long_volume_1024 := tuple(
            trade(index, 0.25, (index % 23) - 11, index % 2 == 0)
            for index in range(1024)
        ),
        _chunks(_long_volume_1024, 4),
    ),
    _scenario(
        "long_257_half_size_events_with_partial_tail",
        2.0,
        _long_volume_257 := tuple(
            trade(index, 0.5, (index % 17) / 2, index % 3 != 0) for index in range(257)
        ),
        _chunks(_long_volume_257, 4)[:-1],
        _chunks(_long_volume_257, 4)[-1],
    ),
)


# Multi-trigger scenarios intentionally mix trigger ordering. Duration is
# checked before admitting an incoming positive-size trade; tick/size checks
# happen after a chunk is added.
MULTI_SCENARIOS: tuple[Scenario[MultiLimits], ...] = (
    _scenario("empty", MultiLimits(1.0, 3, 2.0), ()),
    _scenario(
        "time_only_boundary",
        MultiLimits(1.0, 99, 999.0),
        (t0 := trade(0, 1, 10), t1 := trade(999, 1, 11), t2 := trade(1000, 1, 12)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "tick_only_two_then_partial",
        MultiLimits(100.0, 2, 100.0),
        (t0 := trade(0, 1, 1), t1 := trade(1, 2, 2), t2 := trade(2, 3, 3)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "size_only_exact_threshold",
        MultiLimits(100.0, 99, 2.0),
        (t0 := trade(0, 0.75, 2), t1 := trade(1, 1.25, 4), t2 := trade(2, 0.5, 8)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "max_ticks_one_closes_each_positive_event",
        MultiLimits(100.0, 1, 100.0),
        (t0 := trade(0, 1, 10), trade(1, 0, 11), t1 := trade(2, 1, 12, False)),
        ((t0,), (t1,)),
    ),
    _scenario(
        "just_before_at_and_after_duration",
        MultiLimits(1.0, 99, 100.0),
        (
            t0 := trade(100, 1, 1),
            t1 := trade(1099, 1, 2),
            t2 := trade(1100, 1, 3),
            t3 := trade(1101, 1, 4),
        ),
        ((t0, t1),),
        (t2, t3),
    ),
    _scenario(
        "duration_reanchors_after_gap",
        MultiLimits(1.0, 99, 100.0),
        (
            t0 := trade(0, 1, 1),
            t1 := trade(5000, 1, 2),
            t2 := trade(5999, 1, 3),
            t3 := trade(6000, 1, 4),
        ),
        ((t0,), (t1, t2)),
        (t3,),
    ),
    _scenario(
        "duration_resets_after_tick_close",
        MultiLimits(1.0, 2, 100.0),
        (
            t0 := trade(0, 1, 1),
            t1 := trade(100, 1, 2),
            t2 := trade(200, 1, 3),
            t3 := trade(1199, 1, 4),
            t4 := trade(1200, 1, 5),
        ),
        ((t0, t1), (t2, t3)),
        (t4,),
    ),
    _scenario(
        "duration_resets_after_size_close",
        MultiLimits(1.0, 99, 2.0),
        (
            t0 := trade(0, 1, 1),
            t1 := trade(100, 1, 2),
            t2 := trade(200, 1, 3),
            t3 := trade(1199, 1, 4),
            t4 := trade(1200, 1, 5),
        ),
        ((t0, t1), (t2, t3)),
        (t4,),
    ),
    _scenario(
        "oversized_trade_spans_size_buckets",
        MultiLimits(100.0, 99, 2.0),
        (t0 := trade(5, 5, 4, False),),
        ((p0 := trade(5, 2, 4, False),), (p1 := trade(5, 2, 4, False),)),
        (p2 := trade(5, 1, 4, False),),
    ),
    _scenario(
        "size_close_before_tick_limit",
        MultiLimits(100.0, 5, 2.0),
        (t0 := trade(0, 1, 1), t1 := trade(1, 1, 2), t2 := trade(2, 1, 3)),
        ((t0, t1),),
        (t2,),
    ),
    _scenario(
        "tick_closes_underfilled_then_size_splits",
        MultiLimits(100.0, 2, 10.0),
        (t0 := trade(0, 1, 1), t1 := trade(1, 2, 2, False), t2 := trade(2, 11, 3)),
        ((t0, t1), (p0 := trade(2, 10, 3),)),
        (p1 := trade(2, 1, 3),),
    ),
    _scenario(
        "simultaneous_tick_and_size_closes_once",
        MultiLimits(100.0, 2, 2.0),
        (t0 := trade(0, 1, 1), t1 := trade(1, 1, 2)),
        ((t0, t1),),
    ),
    _scenario(
        "expired_candle_then_large_size_splits",
        MultiLimits(1.0, 99, 2.0),
        (t0 := trade(0, 1, 1), t1 := trade(1000, 5, 2)),
        ((t0,), (p0 := trade(1000, 2, 2),), (p1 := trade(1000, 2, 2),)),
        (p2 := trade(1000, 1, 2),),
    ),
    _scenario(
        "split_fills_existing_candle_then_tick_closes",
        MultiLimits(100.0, 2, 10.0),
        (t0 := trade(0, 1, 1), t1 := trade(1, 20, 2)),
        ((t0, p0 := trade(1, 9, 2)), (p1 := trade(1, 10, 2),)),
        (p2 := trade(1, 1, 2),),
    ),
    _scenario(
        "zero_size_does_not_expire_time_candle",
        MultiLimits(1.0, 99, 100.0),
        (t0 := trade(0, 1, 10), trade(1000, 0, 999), t1 := trade(1001, 1, 11)),
        ((t0,),),
        (t1,),
    ),
    _scenario(
        "zero_size_advances_stale_clock",
        MultiLimits(100.0, 99, 100.0),
        (t0 := trade(10, 1, 10), trade(20, 0, 999), trade(15, 1, -999)),
        current=(t0,),
    ),
    _scenario(
        "duration_priority_preempts_incoming_tick_and_size_tie",
        MultiLimits(1.0, 2, 2.0),
        (t0 := trade(0, 1, 1), t1 := trade(1000, 1, 2, False)),
        ((t0,),),
        (t1,),
    ),
    _scenario(
        "zero_and_negative_prices_with_mixed_sides",
        MultiLimits(100.0, 3, 3.0),
        (t0 := trade(0, 0.5, 0), t1 := trade(1, 1, -2, False), t2 := trade(2, 1.5, 4)),
        ((t0, t1, t2),),
    ),
    _scenario(
        "tiny_sizes_and_threshold",
        MultiLimits(100.0, 99, 1e-13),
        (t0 := trade(0, 2.5e-13, 10),),
        ((p0 := trade(0, 1e-13, 10),), (p1 := trade(0, 1e-13, 10),)),
        (p2 := trade(0, 0.5e-13, 10),),
    ),
    _scenario(
        "stale_after_oversized_trade_ignored",
        MultiLimits(100.0, 99, 2.0),
        (t0 := trade(10, 5, 2), trade(9, 100, 999), t1 := trade(11, 1, 4)),
        (
            (p0 := trade(10, 2, 2),),
            (p1 := trade(10, 2, 2),),
            (p2 := trade(10, 1, 2), t1),
        ),
    ),
    _scenario(
        "same_timestamp_across_completed_chunks",
        MultiLimits(100.0, 99, 1.0),
        (t0 := trade(4, 2.5, 8),),
        ((p0 := trade(4, 1, 8),), (p1 := trade(4, 1, 8),)),
        (p2 := trade(4, 0.5, 8),),
    ),
    _scenario(
        "medium_30_repeated_time_tick_size_winners",
        MultiLimits(5.0, 3, 3.0),
        _medium_multi_cycles_30 := tuple(
            trade(
                timestamp,
                2.5 if index in (6, 13, 20, 27) else 0.5,
                (index % 17) - 8,
                index % 2 == 0,
            )
            for index, timestamp in enumerate(
                (
                    0,
                    1000,
                    5000,
                    6000,
                    7000,
                    8000,
                    9000,
                    10000,
                    11000,
                    15000,
                    16000,
                    17000,
                    18000,
                    19000,
                    20000,
                    21000,
                    25000,
                    26000,
                    27000,
                    28000,
                    29000,
                    30000,
                    31000,
                    35000,
                    36000,
                    37000,
                    38000,
                    39000,
                    40000,
                    41000,
                )
            )
        ),
        (
            _medium_multi_cycles_30[0:2],
            _medium_multi_cycles_30[2:5],
            _medium_multi_cycles_30[5:7],
            _medium_multi_cycles_30[7:9],
            _medium_multi_cycles_30[9:12],
            _medium_multi_cycles_30[12:14],
            _medium_multi_cycles_30[14:16],
            _medium_multi_cycles_30[16:19],
            _medium_multi_cycles_30[19:21],
            _medium_multi_cycles_30[21:23],
            _medium_multi_cycles_30[23:26],
            _medium_multi_cycles_30[26:28],
        ),
        _medium_multi_cycles_30[28:30],
    ),
    _scenario(
        "all_zero_size_events_leave_multi_candle_empty",
        MultiLimits(1.0, 1, 1.0),
        tuple(
            trade(index * 1000, 0, (index % 5) - 2, index % 2 == 0)
            for index in range(16)
        ),
    ),
    _scenario(
        "negative_times_cross_epoch_boundaries",
        MultiLimits(1.0, 99, 100.0),
        (
            t0 := trade(-2000, 1, 1),
            t1 := trade(-1001, 2, 2, False),
            t2 := trade(-1000, 3, 3),
            t3 := trade(-1, 4, 4, False),
            t4 := trade(0, 5, 5),
        ),
        ((t0, t1), (t2, t3)),
        (t4,),
    ),
    _scenario(
        "nextafter_below_size_limit_closes",
        MultiLimits(100.0, 99, 1.0),
        (t0 := trade(0, math.nextafter(1.0, 0.0), 4, False),),
        ((t0,),),
    ),
    _scenario(
        "stale_input_then_exact_boundary_and_equal_time_trade",
        MultiLimits(1.0, 99, 99.0),
        (
            t0 := trade(0, 1, 10),
            trade(-1, 1000, 10000),
            t1 := trade(1000, 1, 11),
            t2 := trade(1000, 2, 12, False),
        ),
        ((t0,),),
        (t1, t2),
    ),
    _scenario(
        "all_sell_split_across_occupied_multi_candle",
        MultiLimits(100.0, 99, 2.0),
        (t0 := trade(0, 0.75, 8, False), t1 := trade(1, 2.5, 4, False)),
        ((t0, p0 := trade(1, 1.25, 4, False)),),
        (p1 := trade(1, 1.25, 4, False),),
    ),
    _scenario(
        "long_257_events_tick_size_and_time_pattern",
        MultiLimits(2.5, 3, 2.0),
        _long_multi_257 := tuple(
            trade(index * 1000, 0.5, (index % 19) - 9, index % 2 == 0)
            for index in range(257)
        ),
        _chunks(_long_multi_257, 3)[:-1],
        _chunks(_long_multi_257, 3)[-1],
    ),
    _scenario(
        "long_1024_quarter_size_groups",
        MultiLimits(1000.0, 9999, 1.0),
        _long_multi_1024 := tuple(
            trade(index, 0.25, (index % 31) - 15, index % 2 == 0)
            for index in range(1024)
        ),
        _chunks(_long_multi_1024, 4),
    ),
)

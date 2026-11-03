"""Shared candle contracts: state, validation, replay, storage and delivery."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from typing import cast

import pytest

from mm_toolbox.candles import (
    MultiCandles,
    PriceCandles,
    TickCandles,
    TimeCandles,
    VolumeCandles,
)
from mm_toolbox.candles.base import BaseCandles, Candle, Trade
from tests.candles.bank import assert_candle, trade

CandleFactory = Callable[[int, bool], BaseCandles]
FACTORIES: tuple[CandleFactory, ...] = (
    lambda n, s: TickCandles(100, n, s),
    lambda n, s: TimeCandles(3600.0, n, s),
    lambda n, s: PriceCandles(1000.0, n, s),
    lambda n, s: VolumeCandles(1000.0, n, s),
    lambda n, s: MultiCandles(3600.0, 100, 1000.0, n, s),
)
CLOSING_FACTORIES: tuple[CandleFactory, ...] = (
    lambda n, s: TickCandles(2, n, s),
    lambda n, s: TimeCandles(1.0, n, s),
    lambda n, s: PriceCandles(1.0, n, s),
    lambda n, s: VolumeCandles(2.0, n, s),
    lambda n, s: MultiCandles(1.0, 2, 2.0, n, s),
)
VARIANT_NAMES = ("tick", "time", "price", "volume", "multi")


@pytest.fixture(params=FACTORIES, ids=VARIANT_NAMES)
def factory(request: pytest.FixtureRequest) -> CandleFactory:
    return cast(CandleFactory, request.param)


@pytest.fixture
def populated_candle() -> Candle:
    return Candle(
        open_time_ms=1000,
        close_time_ms=2000,
        open_price=100.0,
        high_price=102.0,
        low_price=99.0,
        close_price=101.0,
        buy_size=1.0,
        buy_volume=100.0,
        sell_size=2.0,
        sell_volume=202.0,
        vwap=302.0 / 3.0,
        num_trades=2,
        trades=[trade(1000), trade(2000, size=2.0, price=101.0, is_buy=False)],
    )


def test_trade_value_and_immutability() -> None:
    record = trade(1000, size=2.5)
    assert record.value == 250.0
    assert trade(1000, size=0.0).value == 0.0
    with pytest.raises(AttributeError):
        record.price = 99.0  # ty: ignore[invalid-assignment]


def test_empty_candles_have_independent_trade_lists() -> None:
    first, second = Candle.empty(), Candle.empty()
    assert first == second
    assert first.trades is not second.trades
    first.trades.append(trade(0))
    assert second.trades == []


def test_reset_clears_every_field(populated_candle: Candle) -> None:
    populated_candle.reset()
    assert populated_candle == Candle.empty()


@pytest.mark.parametrize("include_trades", [False, True])
def test_copy_preserves_metadata_and_owns_its_trade_list(
    populated_candle: Candle, include_trades: bool
) -> None:
    copied = populated_candle.copy(include_trades=include_trades)
    assert copied is not populated_candle
    assert copied.copy(include_trades=False) == populated_candle.copy(
        include_trades=False
    )
    assert copied.trades is not populated_candle.trades
    if include_trades:
        assert copied.trades == populated_candle.trades
        assert copied.trades[0] is populated_candle.trades[0]
    else:
        assert copied.trades == []
    populated_candle.reset()
    assert copied.num_trades == 2
    assert len(copied.trades) == (2 if include_trades else 0)


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity(factory: CandleFactory, capacity: int) -> None:
    error = ValueError if capacity == 0 else OverflowError
    with pytest.raises(error):
        factory(capacity, True)


@pytest.mark.parametrize("store_trades", [False, True])
def test_shared_aggregation_and_trade_storage(
    factory: CandleFactory, store_trades: bool
) -> None:
    candles = factory(8, store_trades)
    records = (
        trade(1000, size=2.0),
        trade(2000, price=110.0),
        trade(3000, price=90.0, is_buy=False),
    )
    for record in records:
        candles.process_trade(record)
    assert len(candles) == 0
    assert_candle(candles.latest_candle, records, store_trades)


@pytest.mark.parametrize("factory", CLOSING_FACTORIES, ids=VARIANT_NAMES)
@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.asyncio
async def test_closed_history_and_delivery_respect_trade_storage(
    factory: CandleFactory, store_trades: bool
) -> None:
    candles = factory(8, store_trades)
    candles.process_trade(trade(1000))
    candles.process_trade(trade(2000, price=102.0))
    closed = await anext(candles)
    assert closed is candles[0]
    assert len(closed.trades) == (closed.num_trades if store_trades else 0)
    snapshot = closed.copy()
    candles.process_trade(trade(3000, price=103.0))
    assert closed == snapshot


def test_vwap_starts_fresh_after_completion() -> None:
    candles = TickCandles(2)
    candles.process_trade(trade(1000, size=2.0))
    candles.process_trade(trade(2000, price=200.0))
    assert candles[0].vwap == pytest.approx(400.0 / 3.0)
    last = trade(3000, price=50.0, size=4.0)
    candles.process_trade(last)
    assert_candle(candles.latest_candle, (last,))


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_history_indexing_and_iteration_after_wrap(capacity: int) -> None:
    candles = TickCandles(1, num_candles=capacity)
    assert list(candles) == []
    for i in range(20):
        candles.process_trade(trade(i, price=100.0 + i))
    expected = list(range(20 - capacity, 20))
    assert [c.close_time_ms for c in candles] == expected
    assert [candles[i].close_time_ms for i in range(capacity)] == expected
    assert [candles[i - capacity].close_time_ms for i in range(capacity)] == expected
    for index in (capacity, -capacity - 1, 10**100, -(10**100)):
        with pytest.raises(IndexError):
            _ = candles[index]


def test_initialize_replaces_history_and_resets_vwap(factory: CandleFactory) -> None:
    candles = factory(8, True)
    candles.process_trade(trade(10000, price=1000.0, size=50.0))
    records = [trade(1000, price=50.0), trade(2000, price=60.0, is_buy=False)]
    candles.initialize(records)
    assert len(candles) == 0
    assert_candle(candles.latest_candle, tuple(records))


def test_time_reinitialize_resets_trigger_anchor() -> None:
    candles = TimeCandles(10.0)
    candles.initialize([trade(100000), trade(110000, price=101.0)])
    assert len(candles) == 1
    candles.initialize([trade(0), trade(10000, price=110.0)])
    assert len(candles) == 1
    assert_candle(candles[0], (trade(0),))
    assert_candle(candles.latest_candle, (trade(10000, price=110.0),))


@pytest.mark.parametrize("factory", CLOSING_FACTORIES, ids=VARIANT_NAMES)
@pytest.mark.parametrize("store_trades", [False, True])
def test_reinitialize_discards_closed_history_and_trigger_state(
    factory: CandleFactory, store_trades: bool
) -> None:
    candles = factory(8, store_trades)
    candles.initialize([trade(100000), trade(101000, price=102.0)])
    assert len(candles) > 0
    records = [trade(0, price=50.0), trade(1000, price=52.0, is_buy=False)]
    candles.initialize(records)
    fresh = factory(8, store_trades)
    fresh.initialize(records)
    assert list(candles) == list(fresh)
    assert candles.latest_candle == fresh.latest_candle


def test_invalid_history_preserves_state(factory: CandleFactory) -> None:
    candles = factory(8, True)
    candles.process_trade(trade(1000))
    before = candles.latest_candle.copy()
    with pytest.raises(ValueError, match="empty"):
        candles.initialize([])
    with pytest.raises(ValueError, match="Trade"):
        candles.initialize([trade(2000), "invalid"])  # ty: ignore[invalid-argument-type]
    assert candles.latest_candle == before
    candles.process_trade(trade(2000))
    assert candles.latest_candle.num_trades == 2


INVALID_TRADES = (
    pytest.param(trade(3000, size=-1.0), id="negative-size"),
    pytest.param(trade(3000, size=math.nan), id="nan-size"),
    pytest.param(trade(3000, size=math.inf), id="infinite-size"),
    pytest.param(trade(3000, size=-math.inf), id="negative-infinite-size"),
    pytest.param(trade(3000, price=math.nan), id="nan-price"),
    pytest.param(trade(3000, price=math.inf), id="infinite-price"),
    pytest.param(trade(3000, price=-math.inf), id="negative-infinite-price"),
    pytest.param(trade(3000, price=1e308, size=1e308), id="notional-overflow"),
    # Struct constructors intentionally permit injecting invalid annotated fields.
    pytest.param(trade(cast(int, math.nan)), id="nan-time"),
    pytest.param(trade(cast(int, math.inf)), id="infinite-time"),
    pytest.param(trade(cast(int, -math.inf)), id="negative-infinite-time"),
    pytest.param(trade(0, size=-1.0), id="invalid-stale-trade"),
)


@pytest.mark.parametrize("invalid", INVALID_TRADES)
def test_invalid_trade_does_not_change_state_or_advance_time(
    factory: CandleFactory, invalid: Trade
) -> None:
    candles = factory(8, True)
    candles.process_trade(trade(1000))
    snapshot = candles.latest_candle.copy()
    with pytest.raises(ValueError):
        candles.process_trade(invalid)
    assert candles.latest_candle == snapshot
    assert len(candles) == 0
    candles.process_trade(trade(2000, price=101.0))
    assert_candle(candles.latest_candle, (trade(1000), trade(2000, price=101.0)))


@pytest.mark.asyncio
async def test_waiting_consumers_receive_one_broadcast() -> None:
    candles = TickCandles(1)
    assert aiter(candles) is candles
    first = asyncio.create_task(anext(candles))
    peer = asyncio.create_task(anext(candles))
    await asyncio.sleep(0)
    candles.process_trade(trade(1000))
    result = await first
    assert await peer is result
    following = asyncio.create_task(anext(candles))
    await asyncio.sleep(0)
    assert not following.done()
    candles.process_trade(trade(2000))
    assert (await following).close_time_ms == 2000


@pytest.mark.asyncio
async def test_queued_completions_keep_order() -> None:
    candles = TickCandles(1)
    for timestamp in (1000, 2000, 3000):
        candles.process_trade(trade(timestamp))
    assert [(await anext(candles)).close_time_ms for _ in range(3)] == [
        1000,
        2000,
        3000,
    ]


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_lose_completion() -> None:
    candles = TickCandles(1)
    waiter = asyncio.create_task(anext(candles))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    candles.process_trade(trade(1000))
    assert (await anext(candles)).close_time_ms == 1000


@pytest.mark.asyncio
async def test_pending_overrun_is_reported_once_before_retained_delivery() -> None:
    candles = TickCandles(1, num_candles=2)
    for timestamp in (1000, 2000, 3000, 4000):
        candles.process_trade(trade(timestamp))
    with pytest.raises(RuntimeError, match="2 pending candles were dropped"):
        await anext(candles)
    assert (await anext(candles)).close_time_ms == 3000
    assert (await anext(candles)).close_time_ms == 4000
    assert [c.close_time_ms for c in candles] == [3000, 4000]


@pytest.mark.asyncio
async def test_reinitialize_cancels_waiters_and_clears_pending_delivery() -> None:
    candles = TickCandles(1, num_candles=1)
    waiter = asyncio.create_task(anext(candles))
    await asyncio.sleep(0)
    candles.initialize([trade(1000)])
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert (await anext(candles)).close_time_ms == 1000
    candles.process_trade(trade(2000))
    candles.process_trade(trade(3000))
    # Clear both the pending overrun notification and previous closed history.
    candles.initialize([trade(0)])
    assert (await anext(candles)).close_time_ms == 0
    assert len(candles) == 1
    assert candles[0].close_time_ms == 0

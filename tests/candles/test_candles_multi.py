"""Competing duration, tick and size triggers, including split-trade delivery."""

import math

import pytest

import mm_toolbox.candles.multi as multi_module
from mm_toolbox.candles import MultiCandles
from mm_toolbox.candles.base import Trade
from tests.candles.bank import (
    MULTI_SCENARIOS,
    MultiLimits,
    Scenario,
    assert_candle,
    assert_scenario,
    trade,
)


@pytest.mark.parametrize("case", MULTI_SCENARIOS, ids=lambda case: case.name)
@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.parametrize("replay", [False, True], ids=["stream", "replay"])
def test_multi_scenarios(
    case: Scenario[MultiLimits], store_trades: bool, replay: bool
) -> None:
    limit = case.limit
    candles = MultiCandles(
        limit.seconds, limit.ticks, limit.size, max(1, len(case.closed)), store_trades
    )
    if replay:
        if case.trades:
            candles.initialize(list(case.trades))
        else:
            with pytest.raises(ValueError, match="empty"):
                candles.initialize([])
    else:
        for record in case.trades:
            candles.process_trade(record)
    assert_scenario(candles, case, store_trades)


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_duration_and_size_limits(limit: float) -> None:
    with pytest.raises(ValueError, match="max_duration_secs"):
        MultiCandles(limit, 2, 1.0)
    with pytest.raises(ValueError, match="max_size"):
        MultiCandles(1.0, 2, limit)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_tick_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="max_ticks"):
        MultiCandles(1.0, limit, 1.0)


def test_duration_conversion_overflow_raises() -> None:
    with pytest.raises(ValueError, match="max_duration_secs"):
        MultiCandles(1e308, 2, 1.0)


def test_threshold_too_small_to_split_raises() -> None:
    candles = MultiCandles(10.0, 100, 1e-300)
    with pytest.raises(ValueError, match="too small relative"):
        candles.process_trade(trade(1000))


@pytest.mark.parametrize("store_trades", [False, True])
def test_split_trade_allocation_respects_storage(
    monkeypatch: pytest.MonkeyPatch, store_trades: bool
) -> None:
    created: list[Trade] = []

    def counting_trade(time_ms: int, is_buy: bool, price: float, size: float) -> Trade:
        record = Trade(time_ms, is_buy, price, size)
        created.append(record)
        return record

    monkeypatch.setattr(multi_module, "Trade", counting_trade)
    candles = MultiCandles(10.0, 100, 1.0, store_trades=store_trades)
    candles.process_trade(trade(1000, size=2.5))
    assert len(created) == (3 if store_trades else 0)
    for closed in candles:
        assert_candle(closed, (trade(1000),), store_trades)
    assert_candle(candles.latest_candle, (trade(1000, size=0.5),), store_trades)


def test_unsplit_trade_is_stored_by_identity() -> None:
    candles = MultiCandles(10.0, 100, 10.0)
    record = trade(1000)
    candles.process_trade(record)
    assert candles.latest_candle.trades[0] is record


@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.asyncio
async def test_one_oversized_trade_queues_each_closed_chunk(store_trades: bool) -> None:
    candles = MultiCandles(10.0, 10, 1.0, store_trades=store_trades)
    candles.process_trade(trade(1000, size=3.5))
    for i in range(3):
        closed = await anext(candles)
        assert closed is candles[i]
        assert_candle(closed, (trade(1000),), store_trades)
    assert_candle(candles.latest_candle, (trade(1000, size=0.5),), store_trades)

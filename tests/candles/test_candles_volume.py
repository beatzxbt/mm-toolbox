"""Size thresholds, trade splitting and allocation-free aggregation."""

import math

import pytest

import mm_toolbox.candles.volume as volume_module
from mm_toolbox.candles import VolumeCandles
from mm_toolbox.candles.base import Trade
from tests.candles.bank import (
    VOLUME_SCENARIOS,
    Scenario,
    assert_candle,
    assert_scenario,
    trade,
)


@pytest.mark.parametrize("case", VOLUME_SCENARIOS, ids=lambda case: case.name)
@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.parametrize("replay", [False, True], ids=["stream", "replay"])
def test_volume_scenarios(
    case: Scenario[float], store_trades: bool, replay: bool
) -> None:
    candles = VolumeCandles(case.limit, max(1, len(case.closed)), store_trades)
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
def test_invalid_volume_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="volume_per_bucket"):
        VolumeCandles(limit)


def test_threshold_too_small_to_split_raises() -> None:
    candles = VolumeCandles(1e-300)
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

    monkeypatch.setattr(volume_module, "Trade", counting_trade)
    candles = VolumeCandles(1.0, store_trades=store_trades)
    candles.process_trade(trade(1000, size=2.5))
    assert len(created) == (3 if store_trades else 0)
    for closed in candles:
        assert_candle(closed, (trade(1000),), store_trades)
    assert_candle(candles.latest_candle, (trade(1000, size=0.5),), store_trades)


def test_unsplit_trade_is_stored_by_identity() -> None:
    candles = VolumeCandles(10.0)
    record = trade(1000)
    candles.process_trade(record)
    assert candles.latest_candle.trades[0] is record

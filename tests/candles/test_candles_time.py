"""Event-time buckets, boundary ownership, gaps and replay anchoring."""

import math

import pytest

from mm_toolbox.candles import TimeCandles
from tests.candles.bank import TIME_SCENARIOS, Scenario, assert_scenario, trade


@pytest.mark.parametrize("case", TIME_SCENARIOS, ids=lambda case: case.name)
@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.parametrize("replay", [False, True], ids=["stream", "replay"])
def test_time_scenarios(
    case: Scenario[float], store_trades: bool, replay: bool
) -> None:
    candles = TimeCandles(case.limit, max(1, len(case.closed)), store_trades)
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


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf, -math.inf, 1e308])
def test_invalid_time_limit(limit: float) -> None:
    with pytest.raises(ValueError):
        TimeCandles(limit)


def test_interval_below_timestamp_precision_raises() -> None:
    candles = TimeCandles(1e-20)
    with pytest.raises(ValueError, match="timestamp precision"):
        candles.process_trade(trade(1000))

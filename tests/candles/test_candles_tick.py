"""Tick-count triggers across shared deterministic trade scenarios."""

import pytest

from mm_toolbox.candles import TickCandles
from tests.candles.bank import TICK_SCENARIOS, Scenario, assert_scenario


@pytest.mark.parametrize("case", TICK_SCENARIOS, ids=lambda case: case.name)
@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.parametrize("replay", [False, True], ids=["stream", "replay"])
def test_tick_scenarios(case: Scenario[int], store_trades: bool, replay: bool) -> None:
    candles = TickCandles(case.limit, max(1, len(case.closed)), store_trades)
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


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_tick_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="ticks_per_bucket"):
        TickCandles(limit)

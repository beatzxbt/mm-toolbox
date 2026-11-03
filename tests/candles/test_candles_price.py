"""Strict price-movement triggers with explicit closed and live memberships."""

import math

import pytest

from mm_toolbox.candles import PriceCandles
from tests.candles.bank import PRICE_SCENARIOS, Scenario, assert_scenario


@pytest.mark.parametrize("case", PRICE_SCENARIOS, ids=lambda case: case.name)
@pytest.mark.parametrize("store_trades", [False, True])
@pytest.mark.parametrize("replay", [False, True], ids=["stream", "replay"])
def test_price_scenarios(
    case: Scenario[float], store_trades: bool, replay: bool
) -> None:
    candles = PriceCandles(case.limit, max(1, len(case.closed)), store_trades)
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
def test_invalid_price_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="price_bucket"):
        PriceCandles(limit)

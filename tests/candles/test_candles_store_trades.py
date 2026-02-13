"""Tests for configurable trade storage behavior in candle aggregators."""

import asyncio

import pytest

import mm_toolbox.candles.multi as multi_module
import mm_toolbox.candles.volume as volume_module
from mm_toolbox.candles import (
    MultiCandles,
    PriceCandles,
    TickCandles,
    TimeCandles,
    VolumeCandles,
)
from mm_toolbox.candles.base import Candle, Trade as BaseTrade


class TestCandleCopyTrades:
    """Validate Candle.copy(include_trades=...) behavior."""

    def test_copy_include_trades_true_deepcopies(self):
        trade = BaseTrade(time_ms=1000, is_buy=True, price=100.0, size=1.0)
        candle = Candle(
            open_time_ms=1000,
            close_time_ms=2000,
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            buy_size=1.0,
            buy_volume=100.0,
            sell_size=0.0,
            sell_volume=0.0,
            vwap=100.0,
            num_trades=1,
            trades=[trade],
        )

        copied = candle.copy(include_trades=True)

        assert copied is not candle
        assert copied.trades is not candle.trades
        assert copied.trades[0] is not candle.trades[0]
        assert copied.num_trades == candle.num_trades

    def test_copy_include_trades_false_omits_trade_payload(self):
        trade = BaseTrade(time_ms=1000, is_buy=False, price=99.0, size=2.0)
        candle = Candle(
            open_time_ms=1000,
            close_time_ms=2000,
            open_price=100.0,
            high_price=101.0,
            low_price=98.0,
            close_price=99.0,
            buy_size=0.0,
            buy_volume=0.0,
            sell_size=2.0,
            sell_volume=198.0,
            vwap=99.0,
            num_trades=1,
            trades=[trade],
        )

        copied = candle.copy(include_trades=False)

        assert copied is not candle
        assert copied.open_time_ms == candle.open_time_ms
        assert copied.close_time_ms == candle.close_time_ms
        assert copied.vwap == candle.vwap
        assert copied.num_trades == candle.num_trades
        assert copied.trades == []


@pytest.mark.parametrize(
    ("factory", "trade"),
    [
        (
            lambda store_trades: TickCandles(8, store_trades=store_trades),
            BaseTrade(time_ms=1000, is_buy=True, price=100.0, size=1.0),
        ),
        (
            lambda store_trades: TimeCandles(60.0, store_trades=store_trades),
            BaseTrade(time_ms=1000, is_buy=False, price=99.5, size=2.0),
        ),
        (
            lambda store_trades: PriceCandles(1.0, store_trades=store_trades),
            BaseTrade(time_ms=1000, is_buy=True, price=101.0, size=0.5),
        ),
        (
            lambda store_trades: VolumeCandles(10.0, store_trades=store_trades),
            BaseTrade(time_ms=1000, is_buy=False, price=99.0, size=1.5),
        ),
        (
            lambda store_trades: MultiCandles(
                3600.0, 100, 1000.0, store_trades=store_trades
            ),
            BaseTrade(time_ms=1000, is_buy=True, price=100.0, size=1.0),
        ),
    ],
)
def test_store_trades_controls_latest_trade_list(factory, trade):
    with_trades = factory(True)
    without_trades = factory(False)

    with_trades.process_trade(trade)
    without_trades.process_trade(trade)

    assert with_trades.latest_candle.num_trades == 1
    assert without_trades.latest_candle.num_trades == 1
    assert len(with_trades.latest_candle.trades) == 1
    assert len(without_trades.latest_candle.trades) == 0


@pytest.mark.parametrize("store_trades, expected_trade_count", [(True, 1), (False, 0)])
def test_insert_and_reset_copy_respects_store_trades(
    store_trades, expected_trade_count
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        candles = TickCandles(1, store_trades=store_trades)
        trade = BaseTrade(time_ms=1000, is_buy=True, price=100.0, size=1.0)
        candles.process_trade(trade)

        closed_candle = loop.run_until_complete(candles.__anext__())
        assert closed_candle.num_trades == 1
        assert len(closed_candle.trades) == expected_trade_count
    finally:
        loop.close()


def test_volume_split_trade_object_not_created_when_store_trades_false(monkeypatch):
    calls = {"count": 0}

    def counting_trade(*args, **kwargs):
        calls["count"] += 1
        return BaseTrade(*args, **kwargs)

    monkeypatch.setattr(volume_module, "Trade", counting_trade)

    with_trades = VolumeCandles(1.0, store_trades=True)
    without_trades = VolumeCandles(1.0, store_trades=False)
    split_trade = BaseTrade(time_ms=1000, is_buy=True, price=100.0, size=2.5)

    with_trades.process_trade(split_trade)
    with_trades_calls = calls["count"]
    assert with_trades_calls > 0

    calls["count"] = 0
    without_trades.process_trade(split_trade)
    assert calls["count"] == 0
    assert without_trades.latest_candle.trades == []


def test_multi_split_trade_object_not_created_when_store_trades_false(monkeypatch):
    calls = {"count": 0}

    def counting_trade(*args, **kwargs):
        calls["count"] += 1
        return BaseTrade(*args, **kwargs)

    monkeypatch.setattr(multi_module, "Trade", counting_trade)

    with_trades = MultiCandles(3600.0, 100, 1.0, store_trades=True)
    without_trades = MultiCandles(3600.0, 100, 1.0, store_trades=False)
    split_trade = BaseTrade(time_ms=1000, is_buy=False, price=100.0, size=2.5)

    with_trades.process_trade(split_trade)
    with_trades_calls = calls["count"]
    assert with_trades_calls > 0

    calls["count"] = 0
    without_trades.process_trade(split_trade)
    assert calls["count"] == 0
    assert without_trades.latest_candle.trades == []

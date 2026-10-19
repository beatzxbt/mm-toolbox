"""Candle delivery, event-time replay, and quantity-conservation regressions."""

from __future__ import annotations

import asyncio
import math

import pytest

from mm_toolbox.candles import MultiCandles, TickCandles, TimeCandles, VolumeCandles
from mm_toolbox.candles.base import Trade


def trade(timestamp: int, size: float = 1.0, price: float = 100.0) -> Trade:
    """Construct a buy trade for deterministic event-time scenarios."""
    return Trade(time_ms=timestamp, is_buy=True, price=price, size=size)


class TestCandleValidation:
    """Layer 1 — Invalid limits fail before aggregation starts."""

    @pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_limits(self, limit):
        """Nonpositive and nonfinite limits cannot enter processing loops."""
        for factory in (
            lambda: TimeCandles(limit),
            lambda: VolumeCandles(limit),
            lambda: MultiCandles(1.0, 10, limit),
            lambda: MultiCandles(limit, 10, 1.0),
        ):
            with pytest.raises(ValueError):
                factory()

    @pytest.mark.parametrize("size", [-1.0, math.nan, math.inf])
    def test_invalid_trade_preserves_state(self, size):
        """Invalid quantities do not poison the current candle or timestamp."""
        candles = TickCandles(2)
        with pytest.raises(ValueError):
            candles.process_trade(trade(2000, size=size))
        candles.process_trade(trade(1000))
        assert candles.latest_candle.num_trades == 1


class TestCandleReplayAndSplitting:
    """Layer 2 — Resetting buckets preserves stream ordering and quantity."""

    def test_stale_trade_after_close(self):
        """Closing a bucket must not clear the stream's last accepted time."""
        candles = TickCandles(1)
        candles.process_trade(trade(2000))
        candles.process_trade(trade(1000))
        candles.process_trade(trade(2000, price=101.0))
        assert len(candles) == 2
        assert [candles[i].close_time_ms for i in range(2)] == [2000, 2000]
        assert candles[1].close_price == 101.0

    def test_historical_replay_and_reinitialize(self):
        """Replay boundaries depend on first event time and reset on initialize."""
        candles = TimeCandles(10.0)
        history = [trade(1000 + i * 10000, price=100 + i) for i in range(4)]
        candles.initialize(history)
        assert len(candles) == 3
        assert [candles[i].close_price for i in range(3)] == [100, 101, 102]
        assert candles.latest_candle.close_price == 103
        candles.initialize([trade(0), trade(10000, price=110)])
        assert len(candles) == 1
        assert candles[0].open_time_ms == 0
        assert candles.latest_candle.open_time_ms == 10000

    @pytest.mark.parametrize("tiny", [False, True])
    @pytest.mark.parametrize("multi", [False, True])
    def test_split_conserves_stored_trade_quantities(self, tiny, multi):
        """Splits terminate below the old epsilon and retain correct last chunks."""
        threshold = 1e-13 if tiny else 1.0
        size = threshold * 2.5
        candles = (
            MultiCandles(10.0, 100, threshold) if multi else VolumeCandles(threshold)
        )
        candles.process_trade(trade(1000, size=size))
        buckets = [candles[i] for i in range(len(candles))] + [candles.latest_candle]
        assert len(candles) == 2
        assert math.fsum(c.buy_size for c in buckets) == pytest.approx(
            size, rel=1e-14, abs=0
        )
        assert math.fsum(c.buy_volume for c in buckets) == pytest.approx(
            size * 100, rel=1e-14, abs=0
        )
        assert math.fsum(t.size for c in buckets for t in c.trades) == pytest.approx(
            size, rel=1e-14, abs=0
        )
        assert all(c.num_trades == 1 for c in buckets)


class TestCandleAsyncDelivery:
    """Layer 3 — Waiting consumers and queued completions see no duplicates."""

    @pytest.mark.asyncio
    async def test_waiting_broadcast_is_not_delivered_twice(self):
        """Both waiting consumers get one closure; their next wait needs new data."""
        candles = TickCandles(1)
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
    async def test_queued_completions_keep_order(self):
        """Several closures before a consumer runs remain independently pending."""
        candles = TickCandles(1)
        for timestamp in [1000, 2000, 3000]:
            candles.process_trade(trade(timestamp))
        assert [(await anext(candles)).close_time_ms for _ in range(3)] == [
            1000,
            2000,
            3000,
        ]

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_lose_completion(self):
        """A cancelled future is removed and the next closure stays pending."""
        candles = TickCandles(1)
        waiter = asyncio.create_task(anext(candles))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        candles.process_trade(trade(1000))
        assert (await anext(candles)).close_time_ms == 1000

    @pytest.mark.asyncio
    async def test_pending_overrun_is_observable(self):
        """Bounded delivery storage reports loss before yielding retained data."""
        candles = TickCandles(1, num_candles=1)
        candles.process_trade(trade(1000))
        candles.process_trade(trade(2000))
        with pytest.raises(RuntimeError, match="overrun"):
            await anext(candles)
        assert (await anext(candles)).close_time_ms == 2000

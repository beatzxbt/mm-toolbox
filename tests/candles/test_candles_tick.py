"""Layer 2 — Component tests for ``TickCandles``.

Validates tick-count-based candle completion: candles close after a fixed
number of trades regardless of price, volume, or time. Covers exact-count
boundaries, mixed trade sizes (which must not affect the count), rapid
successions, and price-pattern independence.
"""

import asyncio

import pytest

from mm_toolbox.candles import TickCandles
from mm_toolbox.candles.base import Trade


class TestTickCandlesSpecific:
    """Layer 2 — ``TickCandles``-specific ``process_trade`` behaviour."""

    def setup_method(self):
        """Create a fresh asyncio event loop for each test method."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_tick_based_candle_completion(self):
        """Given a bucket of 3 ticks, the 4th trade closes the candle."""
        tick_candles = TickCandles(3)

        for i in range(3):
            trade = Trade(
                time_ms=1640995200000 + i * 1000, is_buy=True, price=100.0 + i, size=1.0
            )
            tick_candles.process_trade(trade)

        final_trade = Trade(time_ms=1640995203000, is_buy=True, price=110.0, size=1.0)
        tick_candles.process_trade(final_trade)

        assert len(tick_candles) == 1
        assert tick_candles.latest_candle.num_trades == 1
        assert tick_candles.latest_candle.open_price == 110.0

    def test_tick_count_accuracy(self):
        """Given various bucket sizes, exactly *count* trades close one candle."""
        for count in [1, 3, 5, 7, 10, 15]:
            tc = TickCandles(count)

            for i in range(count):
                trade = Trade(
                    time_ms=1640995200000 + i * 1000,
                    is_buy=i % 2 == 0,
                    price=100.0 + i,
                    size=1.0,
                )
                tc.process_trade(trade)

            assert len(tc) == 1
            assert tc.latest_candle.num_trades == 0

    def test_tick_candles_with_mixed_trade_sizes(self):
        """Given trades with very different sizes, the tick count is unaffected."""
        tick_candles = TickCandles(4)

        trades = [
            Trade(time_ms=1000, is_buy=True, price=100.0, size=0.1),
            Trade(time_ms=2000, is_buy=False, price=99.0, size=10.0),
            Trade(time_ms=3000, is_buy=True, price=101.0, size=0.001),
            Trade(time_ms=4000, is_buy=False, price=98.0, size=1000.0),
        ]

        for trade in trades:
            tick_candles.process_trade(trade)

        assert len(tick_candles) == 1
        assert tick_candles.latest_candle.num_trades == 0

    def test_tick_candles_rapid_succession(self):
        """Given 25 trades in 1ms intervals, exactly 2 candles complete with 5 remaining."""
        tick_candles = TickCandles(10)

        for i in range(25):
            trade = Trade(
                time_ms=1640995200000 + i,
                is_buy=i % 3 == 0,
                price=100.0 + (i * 0.01),
                size=1.0 + (i * 0.1),
            )
            tick_candles.process_trade(trade)

        assert len(tick_candles) == 2
        assert tick_candles.latest_candle.num_trades == 5

    def test_tick_candles_price_patterns(self):
        """Given up, down, volatile, and flat patterns, tick counting is invariant."""
        price_patterns = [
            [100.0, 101.0, 102.0, 103.0],
            [100.0, 99.0, 98.0, 97.0],
            [100.0, 102.0, 98.0, 101.0],
            [100.0, 100.0, 100.0, 100.0],
        ]

        for pattern in price_patterns:
            tc = TickCandles(len(pattern))
            for i, price in enumerate(pattern):
                trade = Trade(
                    time_ms=1640995200000 + i * 1000,
                    is_buy=i % 2 == 0,
                    price=price,
                    size=1.0,
                )
                tc.process_trade(trade)

            assert len(tc) == 1
            assert tc.latest_candle.num_trades == 0

    def test_invalid_ticks_per_bucket(self):
        """Given a non-positive tick count, construction raises ValueError."""
        with pytest.raises(ValueError):
            TickCandles(0)
        with pytest.raises(ValueError):
            TickCandles(-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

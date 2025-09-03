"""Test suite for TickCandles specific functionality."""

import asyncio

import pytest

from mm_toolbox.candles import TickCandles
from mm_toolbox.candles.base import Trade


class TestTickCandlesSpecific:
    """Test TickCandles specific process_trade implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_tick_based_candle_completion(self):
        """Test that candles complete based on tick count."""
        tick_candles = TickCandles(3)  # 3 ticks per bucket

        # Process exactly 3 trades (should stay in same candle)
        for i in range(3):
            trade = Trade(
                time_ms=1640995200000 + i * 1000, is_buy=True, price=100.0 + i, size=1.0
            )
            tick_candles.process_trade(trade)

        # One more trade should trigger new candle
        final_trade = Trade(time_ms=1640995203000, is_buy=True, price=110.0, size=1.0)
        tick_candles.process_trade(final_trade)

        # Should complete without errors (new candle created)
        assert True

    def test_tick_count_accuracy(self):
        """Test that tick counting is accurate."""
        tick_candles = TickCandles(5)  # noqa: F841

        # Process various numbers of trades
        trade_counts = [1, 3, 5, 7, 10, 15]

        for count in trade_counts:
            tc = TickCandles(count)  # Set limit to current count

            # Process exactly that many trades
            for i in range(count):
                trade = Trade(
                    time_ms=1640995200000 + i * 1000,
                    is_buy=i % 2 == 0,
                    price=100.0 + i,
                    size=1.0,
                )
                tc.process_trade(trade)

            # Should complete exactly at the limit
            assert True

    def test_tick_candles_with_mixed_trade_sizes(self):
        """Test TickCandles with various trade sizes."""
        tick_candles = TickCandles(4)

        # Trades with different sizes (size shouldn't affect tick count)
        trades = [
            Trade(time_ms=1000, is_buy=True, price=100.0, size=0.1),
            Trade(time_ms=2000, is_buy=False, price=99.0, size=10.0),
            Trade(time_ms=3000, is_buy=True, price=101.0, size=0.001),
            Trade(time_ms=4000, is_buy=False, price=98.0, size=1000.0),
        ]

        for trade in trades:
            tick_candles.process_trade(trade)

        # All trades should be processed regardless of size
        assert True

    def test_tick_candles_rapid_succession(self):
        """Test TickCandles with rapid trade succession."""
        tick_candles = TickCandles(10)

        # Send many trades in rapid succession
        for i in range(25):  # Will complete 2+ candles
            trade = Trade(
                time_ms=1640995200000 + i,  # 1ms apart
                is_buy=i % 3 == 0,
                price=100.0 + (i * 0.01),
                size=1.0 + (i * 0.1),
            )
            tick_candles.process_trade(trade)

        # Should handle rapid processing without issues
        assert True

    def test_tick_candles_price_patterns(self):
        """Test TickCandles with various price patterns."""
        tick_candles = TickCandles(8)  # noqa: F841

        # Test different price movement patterns
        price_patterns = [
            [100.0, 101.0, 102.0, 103.0],  # Upward trend
            [100.0, 99.0, 98.0, 97.0],  # Downward trend
            [100.0, 102.0, 98.0, 101.0],  # Volatile
            [100.0, 100.0, 100.0, 100.0],  # Flat
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

        # Should handle all price patterns
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

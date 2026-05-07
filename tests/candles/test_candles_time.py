"""Test suite for TimeCandles specific functionality."""

import asyncio

import pytest

from mm_toolbox.candles import TimeCandles
from mm_toolbox.candles.base import Trade
from mm_toolbox.time.time import time_ms


class TestTimeCandlesSpecific:
    """Test TimeCandles specific process_trade implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_time_based_candle_completion(self):
        """Test that candles complete based on time duration."""
        time_candles = TimeCandles(60.0)  # 60 seconds per bucket
        base_time = time_ms()

        # First trade
        trade1 = Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0)
        time_candles.process_trade(trade1)

        # Trade within duration (30 seconds later)
        trade2 = Trade(time_ms=base_time + 30000, is_buy=True, price=101.0, size=1.0)
        time_candles.process_trade(trade2)

        # Trade beyond duration (70 seconds from start)
        trade3 = Trade(time_ms=base_time + 70000, is_buy=True, price=102.0, size=1.0)
        time_candles.process_trade(trade3)

        assert len(time_candles) == 1
        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.open_price == 102.0

    def test_time_boundary_precision(self):
        """Test precision at time boundaries."""
        time_candles = TimeCandles(30.0)  # 30 seconds
        base_time = time_ms()

        # Test exactly at boundary
        boundary_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(
                time_ms=base_time + 29999, is_buy=True, price=101.0, size=1.0
            ),  # Just before
            Trade(
                time_ms=base_time + 30000, is_buy=True, price=102.0, size=1.0
            ),  # Exactly at
            Trade(
                time_ms=base_time + 30001, is_buy=True, price=103.0, size=1.0
            ),  # Just after
        ]

        for trade in boundary_trades:
            time_candles.process_trade(trade)

        assert len(time_candles) >= 1
        assert time_candles.latest_candle.num_trades >= 1

    def test_late_first_trade_no_empty_candle(self):
        """Ensure a late first trade does not insert an empty candle."""
        time_candles = TimeCandles(60.0)
        # Trade arrives well after construction time
        late_time = time_ms() + 120000
        late_trade = Trade(time_ms=late_time, is_buy=True, price=100.0, size=1.0)
        time_candles.process_trade(late_trade)

        assert len(time_candles) == 0
        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.open_price == 100.0

    def test_simultaneous_timestamp_handling(self):
        """Test handling of trades with identical timestamps."""
        time_candles = TimeCandles(120.0)  # 2 minutes
        same_time = time_ms()

        # Multiple trades at exact same timestamp
        simultaneous_trades = [
            Trade(time_ms=same_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=same_time, is_buy=False, price=99.5, size=2.0),
            Trade(time_ms=same_time, is_buy=True, price=100.5, size=1.5),
            Trade(time_ms=same_time, is_buy=False, price=99.8, size=0.8),
        ]

        for trade in simultaneous_trades:
            time_candles.process_trade(trade)

        assert time_candles.latest_candle.num_trades == 4
        assert time_candles.latest_candle.high_price == 100.5
        assert time_candles.latest_candle.low_price == 99.5

    def test_time_sequence_validation(self):
        """Test that time-based logic works with proper sequences."""
        time_candles = TimeCandles(10.0)  # 10 seconds
        base_time = time_ms()

        # Send trades in proper time sequence
        for i in range(5):
            trade = Trade(
                time_ms=base_time + i * 5000,  # 5 seconds apart
                is_buy=i % 2 == 0,
                price=100.0 + i,
                size=1.0,
            )
            time_candles.process_trade(trade)

        assert time_candles.latest_candle.num_trades >= 1

    def test_time_candles_with_gaps(self):
        """Test TimeCandles with time gaps between trades."""
        time_candles = TimeCandles(60.0)  # 1 minute
        base_time = time_ms()

        # Trades with various time gaps
        gap_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(
                time_ms=base_time + 10000, is_buy=True, price=101.0, size=1.0
            ),  # 10s gap
            Trade(
                time_ms=base_time + 45000, is_buy=False, price=99.0, size=1.0
            ),  # 35s gap
            Trade(
                time_ms=base_time + 65000, is_buy=True, price=102.0, size=1.0
            ),  # 20s gap, new candle
        ]

        for trade in gap_trades:
            time_candles.process_trade(trade)

        assert len(time_candles) == 1
        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.open_price == 102.0

    def test_long_duration_candles(self):
        """Test TimeCandles with longer durations."""
        time_candles = TimeCandles(300.0)  # 5 minutes
        base_time = time_ms()

        # Send trades over several minutes
        for i in range(20):
            trade = Trade(
                time_ms=base_time + i * 10000,  # 10 seconds apart
                is_buy=i % 3 == 0,
                price=100.0 + (i * 0.1),
                size=1.0,
            )
            time_candles.process_trade(trade)

        # Total time: 20 * 10s = 200s < 300s (should stay in same candle)
        assert len(time_candles) == 0
        assert time_candles.latest_candle.num_trades == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

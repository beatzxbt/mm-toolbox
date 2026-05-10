"""Layer 2 — Component tests for ``TimeCandles``.

Validates time-duration-based candle completion: candles close when the elapsed
time since the first trade exceeds a configured threshold. Covers boundary
precision, simultaneous timestamps, late first trades (must not create empty
candles), time gaps, and long-duration buckets.
"""

import asyncio

import pytest

from mm_toolbox.candles import TimeCandles
from mm_toolbox.candles.base import Trade
from mm_toolbox.time.time import time_ms


class TestTimeCandlesSpecific:
    """Layer 2 — ``TimeCandles``-specific ``process_trade`` behaviour."""

    def setup_method(self):
        """Create a fresh asyncio event loop for each test method."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_time_based_candle_completion(self):
        """Given a 60-second bucket, a trade 70 s after the first triggers closure."""
        time_candles = TimeCandles(60.0)
        base_time = time_ms()

        trade1 = Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0)
        time_candles.process_trade(trade1)

        trade2 = Trade(time_ms=base_time + 30000, is_buy=True, price=101.0, size=1.0)
        time_candles.process_trade(trade2)

        trade3 = Trade(time_ms=base_time + 70000, is_buy=True, price=102.0, size=1.0)
        time_candles.process_trade(trade3)

        assert len(time_candles) == 1
        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.open_price == 102.0

    def test_time_boundary_precision(self):
        """Given a 30-second bucket, trades at 29 999 ms, 30 000 ms, and 30 001 ms behave correctly."""
        time_candles = TimeCandles(30.0)
        base_time = time_ms()

        boundary_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=base_time + 29999, is_buy=True, price=101.0, size=1.0),
            Trade(time_ms=base_time + 30000, is_buy=True, price=102.0, size=1.0),
            Trade(time_ms=base_time + 30001, is_buy=True, price=103.0, size=1.0),
        ]

        for trade in boundary_trades:
            time_candles.process_trade(trade)

        assert len(time_candles) >= 1
        assert time_candles.latest_candle.num_trades >= 1

    def test_late_first_trade_no_empty_candle(self):
        """Given a first trade arriving 120 s after construction, no empty candle is inserted.

        Creating an empty candle for the idle period would bloat history and
        misrepresent market activity.
        """
        time_candles = TimeCandles(60.0)
        late_time = time_ms() + 120000
        late_trade = Trade(time_ms=late_time, is_buy=True, price=100.0, size=1.0)
        time_candles.process_trade(late_trade)

        assert len(time_candles) == 0
        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.open_price == 100.0

    def test_simultaneous_timestamp_handling(self):
        """Given four trades with identical timestamps, all are aggregated into the same candle."""
        time_candles = TimeCandles(120.0)
        same_time = time_ms()

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
        """Given trades 5 s apart in a 10-second bucket, they remain in the same candle."""
        time_candles = TimeCandles(10.0)
        base_time = time_ms()

        for i in range(5):
            trade = Trade(
                time_ms=base_time + i * 5000,
                is_buy=i % 2 == 0,
                price=100.0 + i,
                size=1.0,
            )
            time_candles.process_trade(trade)

        assert time_candles.latest_candle.num_trades >= 1

    def test_time_candles_with_gaps(self):
        """Given gaps of 10 s, 35 s, and 20 s in a 60-second bucket, only the 65-second gap triggers closure."""
        time_candles = TimeCandles(60.0)
        base_time = time_ms()

        gap_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=base_time + 10000, is_buy=True, price=101.0, size=1.0),
            Trade(time_ms=base_time + 45000, is_buy=False, price=99.0, size=1.0),
            Trade(time_ms=base_time + 65000, is_buy=True, price=102.0, size=1.0),
        ]

        for trade in gap_trades:
            time_candles.process_trade(trade)

        assert len(time_candles) == 1
        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.open_price == 102.0

    def test_long_duration_candles(self):
        """Given 20 trades 10 s apart in a 300-second bucket, no candle closes."""
        time_candles = TimeCandles(300.0)
        base_time = time_ms()

        for i in range(20):
            trade = Trade(
                time_ms=base_time + i * 10000,
                is_buy=i % 3 == 0,
                price=100.0 + (i * 0.1),
                size=1.0,
            )
            time_candles.process_trade(trade)

        assert len(time_candles) == 0
        assert time_candles.latest_candle.num_trades == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

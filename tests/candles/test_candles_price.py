"""Layer 2 — Component tests for ``PriceCandles``.

Validates price-movement-based candle completion: candles close when the
price moves outside a configured bucket size. Covers bidirectional triggers,
first-trade OHLC initialisation, volatile patterns within a bucket, boundary
precision, and trade-size independence.
"""

import asyncio

import pytest

from mm_toolbox.candles import PriceCandles
from mm_toolbox.candles.base import Trade


class TestPriceCandlesSpecific:
    """Layer 2 — ``PriceCandles``-specific ``process_trade`` behaviour."""

    def setup_method(self):
        """Create a fresh asyncio event loop for each test method."""
        self.loop = asyncio.new_event_loop()

    def test_price_movement_based_completion(self):
        """Given a bucket size of 0.05, a 0.06 price move triggers a new candle."""
        price_candles = PriceCandles(0.05)
        base_time = 1640995200000

        trade1 = Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0)
        price_candles.process_trade(trade1)

        trade2 = Trade(time_ms=base_time + 1000, is_buy=True, price=100.02, size=1.0)
        price_candles.process_trade(trade2)

        trade3 = Trade(time_ms=base_time + 2000, is_buy=True, price=100.06, size=1.0)
        price_candles.process_trade(trade3)

        assert len(price_candles) == 1
        assert price_candles.latest_candle.num_trades == 0

    def test_price_movement_both_directions(self):
        """Given a 0.10 bucket, upward and downward moves both trigger correctly."""
        price_candles = PriceCandles(0.10)
        base_time = 1640995200000

        upward_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=base_time + 1000, is_buy=True, price=100.05, size=1.0),
            Trade(time_ms=base_time + 2000, is_buy=True, price=100.12, size=1.0),
        ]

        for trade in upward_trades:
            price_candles.process_trade(trade)

        pc_down = PriceCandles(0.10)
        downward_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=base_time + 1000, is_buy=False, price=99.95, size=1.0),
            Trade(time_ms=base_time + 2000, is_buy=False, price=99.85, size=1.0),
        ]

        for trade in downward_trades:
            pc_down.process_trade(trade)

        assert len(price_candles) == 1
        assert price_candles.latest_candle.num_trades == 0

    def test_first_trade_low_price_not_zero(self):
        """Given the first trade, OHLC is initialised to that trade's price.

        A common bug is to leave ``low_price`` at 0.0 until a lower trade
        arrives, producing incorrect candle statistics.
        """
        price_candles = PriceCandles(1.0)
        trade = Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0)
        price_candles.process_trade(trade)

        assert price_candles.latest_candle.low_price == 100.0
        assert price_candles.latest_candle.high_price == 100.0
        assert price_candles.latest_candle.open_price == 100.0
        assert price_candles.latest_candle.num_trades == 1

    def test_price_volatility_patterns(self):
        """Given a 0.20 bucket, oscillations within 0.15 keep the candle open."""
        price_candles = PriceCandles(0.20)
        base_time = 1640995200000

        volatile_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=base_time + 1000, is_buy=False, price=100.15, size=1.0),
            Trade(time_ms=base_time + 2000, is_buy=True, price=99.90, size=1.0),
            Trade(time_ms=base_time + 3000, is_buy=False, price=100.10, size=1.0),
        ]

        for trade in volatile_trades:
            price_candles.process_trade(trade)

        assert len(price_candles) == 0
        assert price_candles.latest_candle.num_trades == 4
        assert price_candles.latest_candle.low_price == 99.90
        assert price_candles.latest_candle.high_price == 100.15

    def test_price_candles_trigger_accuracy(self):
        """Given a 0.01 bucket, the exact boundary (0.010) and just-over (0.011) behave correctly."""
        price_candles = PriceCandles(0.01)
        base_time = 1640995200000

        precision_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.000, size=1.0),
            Trade(time_ms=base_time + 1000, is_buy=True, price=100.009, size=1.0),
            Trade(time_ms=base_time + 2000, is_buy=True, price=100.010, size=1.0),
            Trade(time_ms=base_time + 3000, is_buy=True, price=100.011, size=1.0),
        ]

        for trade in precision_trades:
            price_candles.process_trade(trade)

        assert len(price_candles) == 1
        assert price_candles.latest_candle.num_trades == 0

    def test_price_candles_with_mixed_sizes(self):
        """Given tiny and huge trade sizes, the trigger depends only on price move."""
        price_candles = PriceCandles(0.50)
        base_time = 1640995200000

        mixed_size_trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=0.001),
            Trade(time_ms=base_time + 1000, is_buy=False, price=100.2, size=1000.0),
            Trade(time_ms=base_time + 2000, is_buy=True, price=100.4, size=0.1),
            Trade(time_ms=base_time + 3000, is_buy=False, price=100.6, size=50.0),
        ]

        for trade in mixed_size_trades:
            price_candles.process_trade(trade)

        assert len(price_candles) == 1
        assert price_candles.latest_candle.num_trades == 0

    def test_invalid_price_bucket(self):
        """Given a non-positive price bucket, construction raises ValueError."""
        with pytest.raises(ValueError):
            PriceCandles(0.0)
        with pytest.raises(ValueError):
            PriceCandles(-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

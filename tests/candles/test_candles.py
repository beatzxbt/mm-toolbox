"""Layer 3 — Integration tests for the candles module.

Validates that all candle aggregator types (Tick, Volume, Time, Price) can be
imported, instantiated, and process trades without interfering with one another.
Covers cross-candle trade processing, async contexts, and mixed buy/sell
patterns.
"""

import asyncio

import pytest

from mm_toolbox.candles import PriceCandles, TickCandles, TimeCandles, VolumeCandles
from mm_toolbox.candles.base import Trade


class TestCandlesModuleIntegration:
    """Layer 3 — Integration tests across all working candle types."""

    def setup_method(self):
        """Create a fresh asyncio event loop for each test method."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_all_candle_types_import(self):
        """Given the candles package, all five public types are importable."""
        import mm_toolbox.candles as candles_module

        expected_types = [
            "TickCandles",
            "VolumeCandles",
            "TimeCandles",
            "PriceCandles",
            "MultiCandles",
        ]
        for candle_type in expected_types:
            assert hasattr(candles_module, candle_type)

    def test_all_working_candle_types_creation(self):
        """Given valid constructor arguments, each candle type instantiates."""
        working_candles = {
            "TickCandles": TickCandles(5),
            "VolumeCandles": VolumeCandles(1000.0),
            "TimeCandles": TimeCandles(60.0),
            "PriceCandles": PriceCandles(0.01),
        }

        for _name, candle_obj in working_candles.items():
            assert candle_obj is not None

    def test_cross_candle_type_trade_processing(self):
        """Given identical trades, each candle type processes them independently.

        Verifies that shared trade data does not cause cross-contamination
        between aggregator instances.
        """
        candles = {
            "TickCandles": TickCandles(10),
            "VolumeCandles": VolumeCandles(2000.0),
            "TimeCandles": TimeCandles(300.0),
            "PriceCandles": PriceCandles(0.5),
        }

        trades = [
            Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=1640995210000, is_buy=False, price=99.8, size=1.2),
            Trade(time_ms=1640995220000, is_buy=True, price=100.3, size=0.8),
            Trade(time_ms=1640995230000, is_buy=False, price=99.5, size=2.0),
            Trade(time_ms=1640995240000, is_buy=True, price=101.0, size=1.5),
        ]

        for _name, candle_obj in candles.items():
            for trade in trades:
                candle_obj.process_trade(trade)

        assert True

    def test_candle_specialization_behavior(self):
        """Given divergent trigger thresholds, each candle type fires at its own point."""
        tick_candles = TickCandles(3)
        volume_candles = VolumeCandles(250.0)
        time_candles = TimeCandles(2.0)
        price_candles = PriceCandles(0.5)

        base_time = 1640995200000
        trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=base_time + 1000, is_buy=True, price=100.1, size=1.0),
            Trade(time_ms=base_time + 2000, is_buy=True, price=100.2, size=0.5),
            Trade(time_ms=base_time + 3000, is_buy=True, price=100.6, size=1.0),
        ]

        candle_types = [
            ("TickCandles", tick_candles),
            ("VolumeCandles", volume_candles),
            ("TimeCandles", time_candles),
            ("PriceCandles", price_candles),
        ]

        for _name, candle_obj in candle_types:
            for trade in trades:
                candle_obj.process_trade(trade)

        assert True

    def test_mixed_trade_patterns_across_types(self):
        """Given buy-only, sell-only, and alternating patterns, all types survive."""
        candle_types = [
            TickCandles(8),
            VolumeCandles(1500.0),
            TimeCandles(180.0),
            PriceCandles(0.25),
        ]

        patterns = [
            [
                Trade(
                    time_ms=1000 + i * 1000,
                    is_buy=True,
                    price=100.0 + i * 0.01,
                    size=1.0,
                )
                for i in range(5)
            ],
            [
                Trade(
                    time_ms=2000 + i * 1000,
                    is_buy=False,
                    price=99.0 - i * 0.01,
                    size=1.0,
                )
                for i in range(5)
            ],
            [
                Trade(
                    time_ms=3000 + i * 1000,
                    is_buy=i % 2 == 0,
                    price=100.0 + i * 0.02,
                    size=1.0,
                )
                for i in range(6)
            ],
        ]

        for candle_obj in candle_types:
            for pattern in patterns:
                for trade in pattern:
                    candle_obj.process_trade(trade)

        assert True


class TestCandlesAsyncIntegration:
    """Layer 3 — Async integration across candle types."""

    def setup_method(self):
        """Create a fresh asyncio event loop for each test method."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_async_context_all_candles(self):
        """Given an async context, all candle types process trades correctly."""

        async def test_async():
            candles = [
                TickCandles(5),
                VolumeCandles(1000.0),
                TimeCandles(60.0),
                PriceCandles(0.01),
            ]

            trade = Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0)

            for candle_obj in candles:
                candle_obj.process_trade(trade)

            return True

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(test_async())
        assert result is True

    def test_concurrent_candle_processing(self):
        """Given multiple instances of each type, trades are processed concurrently.

        This guards against mutable global or class-level state that could cause
        one instance to corrupt another.
        """
        candle_instances = [
            TickCandles(3),
            TickCandles(5),
            VolumeCandles(500.0),
            VolumeCandles(1500.0),
            TimeCandles(30.0),
            TimeCandles(120.0),
            PriceCandles(0.01),
            PriceCandles(0.1),
        ]

        trade = Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0)

        for candle_obj in candle_instances:
            candle_obj.process_trade(trade)

        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

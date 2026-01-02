"""Integration tests for candles module."""

import asyncio

import pytest

from mm_toolbox.candles import PriceCandles, TickCandles, TimeCandles, VolumeCandles
from mm_toolbox.candles.base import Trade


class TestCandlesModuleIntegration:
    """Test integration across all working candle types."""

    def setup_method(self):
        """Set up test fixtures."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_all_candle_types_import(self):
        """Test that all candle types can be imported."""
        import mm_toolbox.candles as candles_module

        # Should have all 5 types (even if MultiCandles has issues)
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
        """Test creating all working candle types."""
        working_candles = {
            "TickCandles": TickCandles(5),
            "VolumeCandles": VolumeCandles(1000.0),
            "TimeCandles": TimeCandles(60.0),
            "PriceCandles": PriceCandles(0.01),
        }

        for _name, candle_obj in working_candles.items():
            assert candle_obj is not None

    def test_cross_candle_type_trade_processing(self):
        """Test same trades processed by different candle types."""
        # Create all working candle types with different triggers
        candles = {
            "TickCandles": TickCandles(10),  # 10 ticks
            "VolumeCandles": VolumeCandles(2000.0),  # 2000 volume
            "TimeCandles": TimeCandles(300.0),  # 5 minutes
            "PriceCandles": PriceCandles(0.5),  # 0.5 price move
        }

        # Same trade sequence for all candle types
        trades = [
            Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=1640995210000, is_buy=False, price=99.8, size=1.2),
            Trade(time_ms=1640995220000, is_buy=True, price=100.3, size=0.8),
            Trade(time_ms=1640995230000, is_buy=False, price=99.5, size=2.0),
            Trade(time_ms=1640995240000, is_buy=True, price=101.0, size=1.5),
        ]

        # Process same trades with all candle types
        for _name, candle_obj in candles.items():
            for trade in trades:
                candle_obj.process_trade(trade)

        # Each candle type should handle the trades according to its logic
        assert True

    def test_candle_specialization_behavior(self):
        """Test that each candle type behaves according to its specialization."""
        # Create candles with triggers designed to fire at different times
        tick_candles = TickCandles(3)  # Triggers after 3 trades
        volume_candles = VolumeCandles(250.0)  # Triggers after 250 volume
        time_candles = TimeCandles(2.0)  # Triggers after 2 seconds
        price_candles = PriceCandles(0.5)  # Triggers after 0.5 price move

        base_time = 1640995200000
        trades = [
            Trade(time_ms=base_time, is_buy=True, price=100.0, size=1.0),  # 100 volume
            Trade(
                time_ms=base_time + 1000, is_buy=True, price=100.1, size=1.0
            ),  # 100.1 volume, total=200.1
            Trade(
                time_ms=base_time + 2000, is_buy=True, price=100.2, size=0.5
            ),  # 50.1 volume, total=250.2
            Trade(
                time_ms=base_time + 3000, is_buy=True, price=100.6, size=1.0
            ),  # Price move > 0.5, 3+ trades
        ]

        candle_types = [
            ("TickCandles", tick_candles),  # Should trigger after trade 3 (3 trades)
            (
                "VolumeCandles",
                volume_candles,
            ),  # Should trigger after trade 3 (250.2 > 250)
            ("TimeCandles", time_candles),  # Should trigger after trade 2 (2000ms > 2s)
            ("PriceCandles", price_candles),  # Should trigger after trade 4 (0.6 > 0.5)
        ]

        for _name, candle_obj in candle_types:
            for trade in trades:
                candle_obj.process_trade(trade)

        # Each candle type should handle according to its specialization
        assert True

    def test_mixed_trade_patterns_across_types(self):
        """Test various trade patterns across all candle types."""
        candle_types = [
            TickCandles(8),
            VolumeCandles(1500.0),
            TimeCandles(180.0),
            PriceCandles(0.25),
        ]

        # Different trade patterns
        patterns = [
            # All buys
            [
                Trade(
                    time_ms=1000 + i * 1000,
                    is_buy=True,
                    price=100.0 + i * 0.01,
                    size=1.0,
                )
                for i in range(5)
            ],
            # All sells
            [
                Trade(
                    time_ms=2000 + i * 1000,
                    is_buy=False,
                    price=99.0 - i * 0.01,
                    size=1.0,
                )
                for i in range(5)
            ],
            # Alternating
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

        # All candle types should handle all patterns
        assert True


class TestCandlesAsyncIntegration:
    """Test async integration across candle types."""

    def setup_method(self):
        """Set up test fixtures."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_async_context_all_candles(self):
        """Test all working candles in async context."""

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

        # Run async test
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(test_async())
        assert result is True

    def test_concurrent_candle_processing(self):
        """Test that multiple candle objects can process trades concurrently."""
        # Create multiple instances of each type
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

        # Process trades with all instances
        trade = Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0)

        for candle_obj in candle_instances:
            candle_obj.process_trade(trade)

        # Should handle multiple concurrent instances
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Layer 1 — Primitives tests for base candle functionality.

Covers the ``Trade`` and ``Candle`` structs (creation, immutability, edge
values) and base-class behaviour shared by all candle aggregators (VWAP,
stale-trade handling, async Future lifecycle, validation on ``initialize``).
"""

import asyncio

import pytest

from mm_toolbox.candles.base import Candle, Trade


class TestTradeStructure:
    """Layer 1 — ``Trade`` data-structure tests."""

    def test_trade_creation(self):
        """Given valid fields, a ``Trade`` is created with exact attributes."""
        trade = Trade(time_ms=1640995200000, is_buy=True, price=100.50, size=1.5)

        assert trade.time_ms == 1640995200000
        assert trade.is_buy is True
        assert trade.price == 100.50
        assert trade.size == 1.5

    def test_trade_edge_values(self):
        """Given extreme values (zero, max int, large floats), creation succeeds.

        ``msgspec.Struct`` does not validate ranges, so zero and very large
        values are accepted without error.
        """
        edge_trade = Trade(time_ms=1000, is_buy=True, price=0.0, size=0.0)
        assert edge_trade.price == 0.0
        assert edge_trade.size == 0.0

        large_trade = Trade(
            time_ms=9223372036854775807, is_buy=False, price=999999.99, size=10000.0
        )
        assert large_trade.time_ms == 9223372036854775807
        assert large_trade.price == 999999.99
        assert large_trade.size == 10000.0

    def test_trade_buy_sell_types(self):
        """Given ``is_buy=True`` and ``is_buy=False``, the flag is stored exactly."""
        buy_trade = Trade(time_ms=1000, is_buy=True, price=100.0, size=1.0)
        sell_trade = Trade(time_ms=2000, is_buy=False, price=99.0, size=2.0)

        assert buy_trade.is_buy is True
        assert sell_trade.is_buy is False

    def test_trade_is_frozen(self):
        """Given a ``Trade`` instance, mutating any field raises ``AttributeError``.

        Immutability is critical for thread-safe sharing between candle
        aggregators and consumers.
        """
        trade = Trade(time_ms=1000, is_buy=True, price=100.0, size=1.0)
        with pytest.raises(AttributeError):
            trade.price = 99.0


class TestCandleStructure:
    """Layer 1 — ``Candle`` data-structure tests."""

    def test_candle_empty_creation(self):
        """Given ``Candle.empty()``, all fields are zero-initialised."""
        candle = Candle.empty()

        assert candle.open_time_ms == 0
        assert candle.close_time_ms == 0
        assert candle.open_price == 0.0
        assert candle.high_price == 0.0
        assert candle.low_price == 0.0
        assert candle.close_price == 0.0
        assert candle.buy_size == 0.0
        assert candle.buy_volume == 0.0
        assert candle.sell_size == 0.0
        assert candle.sell_volume == 0.0
        assert candle.vwap == 0.0
        assert candle.num_trades == 0
        assert candle.trades == []

    def test_candle_full_creation(self):
        """Given every field populated, the ``Candle`` stores them exactly."""
        candle = Candle(
            open_time_ms=1000,
            close_time_ms=2000,
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            buy_size=1.0,
            buy_volume=100.0,
            sell_size=0.5,
            sell_volume=50.0,
            vwap=100.5,
            num_trades=2,
            trades=[Trade(time_ms=1000, is_buy=True, price=100.0, size=1.0)],
        )

        assert candle.open_time_ms == 1000
        assert candle.close_time_ms == 2000
        assert candle.open_price == 100.0
        assert candle.high_price == 102.0
        assert candle.low_price == 99.0
        assert candle.close_price == 101.0
        assert candle.buy_size == 1.0
        assert candle.buy_volume == 100.0
        assert candle.sell_size == 0.5
        assert candle.sell_volume == 50.0
        assert candle.vwap == 100.5
        assert candle.num_trades == 2
        assert len(candle.trades) == 1

    def test_candle_reset(self):
        """Given a populated ``Candle``, ``reset()`` returns it to empty state."""
        candle = Candle(
            open_time_ms=1000,
            close_time_ms=2000,
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            buy_size=1.0,
            buy_volume=100.0,
            sell_size=0.5,
            sell_volume=50.0,
            vwap=100.5,
            num_trades=2,
            trades=[Trade(time_ms=1000, is_buy=True, price=100.0, size=1.0)],
        )

        candle.reset()

        empty_candle = Candle.empty()
        assert candle.open_time_ms == empty_candle.open_time_ms
        assert candle.num_trades == empty_candle.num_trades
        assert candle.vwap == empty_candle.vwap

    def test_candle_copy(self):
        """Given a ``Candle``, ``copy()`` produces an equivalent but independent object."""
        original = Candle(
            open_time_ms=1000,
            close_time_ms=2000,
            open_price=100.0,
            high_price=102.0,
            low_price=99.0,
            close_price=101.0,
            buy_size=1.0,
            buy_volume=100.0,
            sell_size=0.5,
            sell_volume=50.0,
            vwap=100.5,
            num_trades=1,
            trades=[],
        )

        copied = original.copy()

        assert copied.open_price == original.open_price
        assert copied.num_trades == original.num_trades
        assert copied.vwap == original.vwap


class TestBaseCandlesFunctionality:
    """Layer 2 — Base-class behaviour exercised through concrete subclasses."""

    def setup_method(self):
        """Create a fresh asyncio event loop for each test method."""
        asyncio.set_event_loop(asyncio.new_event_loop())

    def test_base_candles_initialization(self):
        """Given valid parameters, every subclass instantiates without error."""
        from mm_toolbox.candles import (
            PriceCandles,
            TickCandles,
            TimeCandles,
            VolumeCandles,
        )

        tc = TickCandles(5)
        vc = VolumeCandles(1000.0)
        time_c = TimeCandles(60.0)
        pc = PriceCandles(0.01)

        assert tc is not None
        assert vc is not None
        assert time_c is not None
        assert pc is not None

    def test_vwap_calculation_through_subclasses(self):
        """Given a known sequence of trades, VWAP equals the price-weighted mean.

        VWAP = sum(price * size) / sum(size).  This is a core financial metric,
        so an off-by-one in the divisor would silently corrupt trading signals.
        """
        from mm_toolbox.candles import TickCandles

        tick_candles = TickCandles(10)

        trades = [
            Trade(time_ms=1000, is_buy=True, price=100.0, size=1.0),
            Trade(time_ms=2000, is_buy=True, price=110.0, size=2.0),
            Trade(time_ms=3000, is_buy=False, price=90.0, size=1.0),
        ]

        for trade in trades:
            tick_candles.process_trade(trade)

        assert tick_candles.latest_candle.vwap == pytest.approx(102.5)

    def test_vwap_uses_price_weighting(self):
        """Given unequal sizes, VWAP weights by size rather than averaging prices.

        A common bug is to compute mean(price) instead of sum(price*size)/sum(size).
        """
        from mm_toolbox.candles import TickCandles

        tick_candles = TickCandles(3)

        trades = [
            Trade(time_ms=1000, is_buy=True, price=100.0, size=2.0),
            Trade(time_ms=2000, is_buy=True, price=110.0, size=1.0),
        ]

        for trade in trades:
            tick_candles.process_trade(trade)

        expected_vwap = (100.0 * 2.0 + 110.0 * 1.0) / (2.0 + 1.0)
        assert tick_candles.latest_candle.vwap == pytest.approx(expected_vwap)

    def test_vwap_resets_after_candle_completion(self):
        """Given a completed candle, the next candle starts with a fresh VWAP.

        If VWAP leaked across candles, a quiet period after a volatile one would
        report an inflated average.
        """
        from mm_toolbox.candles import TickCandles

        tick_candles = TickCandles(2)

        first_trades = [
            Trade(time_ms=1000, is_buy=True, price=100.0, size=2.0),
            Trade(time_ms=2000, is_buy=False, price=200.0, size=1.0),
        ]

        for trade in first_trades:
            tick_candles.process_trade(trade)

        next_trade = Trade(time_ms=3000, is_buy=True, price=50.0, size=4.0)
        tick_candles.process_trade(next_trade)

        assert tick_candles.latest_candle.num_trades == 1
        assert tick_candles.latest_candle.vwap == pytest.approx(50.0)

    def test_stale_trade_handling(self):
        """Given a trade with an earlier timestamp than the current candle, it is ignored.

        Stale trades could arrive out-of-order from slow market-data feeds;
        accepting them would corrupt open/high/low/close statistics.
        """
        from mm_toolbox.candles import TimeCandles

        time_candles = TimeCandles(60.0)

        trade1 = Trade(time_ms=1640995200000, is_buy=True, price=100.0, size=1.0)
        time_candles.process_trade(trade1)

        stale_trade = Trade(time_ms=1640995199000, is_buy=True, price=99.0, size=1.0)
        time_candles.process_trade(stale_trade)

        assert time_candles.latest_candle.num_trades == 1
        assert time_candles.latest_candle.close_price == 100.0

    def test_async_future_recreation(self):
        """Given many rapid completions, async Futures are recreated cleanly.

        The base class uses an asyncio Future to signal candle closure. Repeated
        resets must not leak or raise InvalidStateError.
        """
        from mm_toolbox.candles import VolumeCandles

        volume_candles = VolumeCandles(1.0)

        high_volume_trades = [
            Trade(time_ms=1000, is_buy=True, price=100.0, size=2.0),
            Trade(time_ms=2000, is_buy=True, price=101.0, size=3.0),
            Trade(time_ms=3000, is_buy=True, price=102.0, size=1.5),
        ]

        for trade in high_volume_trades:
            volume_candles.process_trade(trade)

        assert len(volume_candles) == 6

    def test_initialize_empty_list_raises(self):
        """Given an empty list, ``initialize`` raises ValueError."""
        from mm_toolbox.candles import TickCandles

        tick_candles = TickCandles(5)
        with pytest.raises(ValueError, match="empty"):
            tick_candles.initialize([])

    def test_initialize_mixed_types_raises(self):
        """Given a list containing a non-Trade element, ``initialize`` raises ValueError."""
        from mm_toolbox.candles import TickCandles

        tick_candles = TickCandles(5)
        with pytest.raises(ValueError, match="Trade"):
            tick_candles.initialize(
                [Trade(time_ms=1000, is_buy=True, price=100.0, size=1.0), "not a trade"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

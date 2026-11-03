"""Shared test data helpers for candle behavior tests."""

from mm_toolbox.candles.base import Trade


def trade(timestamp: int, size: float = 1.0, price: float = 100.0) -> Trade:
    """Construct a buy trade for deterministic event-time scenarios."""
    return Trade(time_ms=timestamp, is_buy=True, price=price, size=size)

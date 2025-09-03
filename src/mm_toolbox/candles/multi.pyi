from __future__ import annotations

from mm_toolbox.candles.base import BaseCandles

__all__ = ["MultiCandles"]

class MultiCandles(BaseCandles):
    """Candle aggregator with time, tick, and volume triggers."""

    def __init__(
        self,
        max_duration_secs: float,
        max_ticks: int,
        max_size: float,
        num_candles: int,
    ) -> None: ...
    def process_trade(
        self, time_ms: float, is_buy: bool, price: float, size: float
    ) -> None: ...

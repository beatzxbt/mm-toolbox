from __future__ import annotations

from mm_toolbox.candles.base import BaseCandles

class TickCandles(BaseCandles):
    """Candle aggregator with fixed-trade-count buckets."""

    def __init__(self, ticks_per_bucket: int, num_candles: int) -> None: ...
    def process_trade(
        self, time_ms: float, is_buy: bool, price: float, size: float
    ) -> None: ...

from __future__ import annotations

from mm_toolbox.candles.base import BaseCandles, Trade

class TickCandles(BaseCandles):
    """Candle aggregator with fixed-trade-count buckets."""

    def __init__(self, ticks_per_bucket: int, num_candles: int = 1000) -> None: ...
    def process_trade(self, trade: Trade) -> None: ...

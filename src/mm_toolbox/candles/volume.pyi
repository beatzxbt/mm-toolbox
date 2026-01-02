from __future__ import annotations

from mm_toolbox.candles.base import BaseCandles, Trade

__all__ = ["VolumeCandles"]

class VolumeCandles(BaseCandles):
    """Candle aggregator with fixed volume buckets."""

    def __init__(self, volume_per_bucket: float, num_candles: int = 1000) -> None: ...
    def process_trade(self, trade: Trade) -> None: ...

from __future__ import annotations

from mm_toolbox.candles.base import BaseCandles

__all__ = ["VolumeCandles"]

class VolumeCandles(BaseCandles):
    """Candle aggregator with fixed volume buckets."""

    def __init__(self, volume_per_bucket: float, num_candles: int) -> None: ...
    def process_trade(
        self, time_ms: float, is_buy: bool, price: float, size: float
    ) -> None: ...

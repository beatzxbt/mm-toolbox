from mm_toolbox.candles.base import BaseCandles

class PriceCandles(BaseCandles):
    """Candle aggregator with price-move triggers."""

    def __init__(self, price_bucket: float, num_candles: int) -> None: ...
    def process_trade(
        self, time_ms: float, is_buy: bool, price: float, size: float
    ) -> None: ...

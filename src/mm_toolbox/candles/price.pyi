from mm_toolbox.candles.base import BaseCandles, Trade

class PriceCandles(BaseCandles):
    """Candle aggregator with price-move triggers."""

    def __init__(self, price_bucket: float, num_candles: int = 1000) -> None: ...
    def process_trade(self, trade: Trade) -> None: ...

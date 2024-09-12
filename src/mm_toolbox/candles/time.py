from .base import BaseCandles


class TimeCandles(BaseCandles):
    def __init__(self, secs_per_bucket: float, num_candles: int) -> None:
        self.millis_per_bucket = secs_per_bucket * 1000.0
        super().__init__(num_candles)

    def process_trade(
        self, timestamp: float, side: bool, price: float, size: float
    ) -> None:
        if self.total_trades == 0.0:
            self.open_timestamp = timestamp
            self.open_price = price

        if self.open_timestamp + self.millis_per_bucket <= timestamp:
            self.insert_candle()
            self.process_trade(timestamp, side, price, size)

        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price

        match side:
            case 0.0:
                self.buy_volume += size

            case 1.0:
                self.sell_volume += size

        self.vwap_price = self.calculate_vwap(price, size)
        self.total_trades += 1.0
        self.close_timestamp = timestamp

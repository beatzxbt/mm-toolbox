from .base import BaseCandles


class TickCandles(BaseCandles):
    def __init__(self, ticks_per_bucket: float, num_candles: int) -> None:
        self.ticks_per_bucket = ticks_per_bucket
        super().__init__(num_candles)

    def process_trade(
        self, timestamp: float, side: bool, price: float, size: float
    ) -> None:
        if self.total_trades == 0.0:
            self.open_timestamp = timestamp
            self.open_price = price

        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price

        match side:
            case 0.0:
                self.buy_volume += size

            case 1.0:
                self.sell_volume += size

        self.vwap_price = self.calculate_vwap(price, size)
        self.total_trades += 1
        self.close_timestamp = timestamp

        if self.total_trades >= self.ticks_per_bucket:
            self.insert_candle(
                open_price=self.open_price,
                high_price=self.high_price,
                low_price=self.low_price,
                close_price=self.close_price,
                buy_volume=self.buy_volume,
                sell_volume=self.sell_volume,
                vwap_price=self.vwap_price,
                total_trades=self.total_trades,
                open_timestamp=self.open_timestamp,
                close_timestamp=self.close_timestamp,
            )

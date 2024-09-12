from .base import BaseCandles


class MultiTriggerCandles(BaseCandles):
    def __init__(
        self,
        max_duration_secs: float,
        max_ticks: float,
        max_volume: float,
        num_candles: int,
    ) -> None:
        self.max_duration_millis = max_duration_secs * 1000.0
        self.max_ticks = max_ticks
        self.max_volume = max_volume
        super().__init__(num_candles)

    def process_trade(
        self, timestamp: float, side: bool, price: float, size: float
    ) -> None:
        if self.total_trades == 0.0:
            self.open_timestamp = timestamp
            self.open_price = price

        if self.open_timestamp + self.max_duration_millis <= timestamp:
            self.insert_candle()
            self.process_trade(timestamp, side, price, size)
            return

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

        if self.total_trades >= self.max_ticks:
            self.insert_candle()
            return

        total_volume = self.buy_volume + self.sell_volume
        if total_volume > self.max_volume:
            remaining_volume = total_volume - self.max_volume

            match side:
                case 0.0:
                    self.buy_volume -= remaining_volume

                case 1.0:
                    self.sell_volume -= remaining_volume

            self.insert_candle()
            self.process_trade(timestamp, side, price, remaining_volume)
            return

from libc.stdint cimport uint32_t
from .base cimport BaseCandles


class TimeCandles(BaseCandles):
    def __init__(self, double secs_per_bucket, uint32_t num_candles):
        super().__init__(num_candles)

        self.millis_per_bucket = secs_per_bucket * 1000.0

    cpdef void process_trade(
        self, double time, bint is_buy, double px, double sz
    ):
        if self.total_trades == 0.0:
            self.open_time = time
            self.open_price = px

        if self.open_time + self.millis_per_bucket <= time:
            self.insert_candle()
            self.process_trade(time, is_buy, px, sz)

        self.high_price = max(self.high_price, px)
        self.low_price = min(self.low_price, px)
        self.close_price = px

        if is_buy == 0.0:
            self.buy_volume += sz
        else:
            self.sell_volume += sz

        self.vwap_price = self.calculate_vwap(px, sz)
        self.total_trades += 1.0
        self.close_time = time

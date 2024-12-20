from libc.stdint cimport uint32_t
from .base cimport BaseCandles

class TickCandles(BaseCandles):
    def __init__(self, double ticks_per_bucket, uint32_t num_candles):
        super().__init__(num_candles)
        
        self.ticks_per_bucket = ticks_per_bucket

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz):
        if self.total_trades == 0.0:
            self.open_time = time
            self.open_price = px

        self.high_price = max(self.high_price, px)
        self.low_price = min(self.low_price, px)
        self.close_price = px

        if is_buy == 0.0:
            self.buy_volume += sz
        else:
            self.sell_volume += sz

        self.vwap_price = self.calculate_vwap(px, sz)
        self.total_trades += 1
        self.close_time = time

        if self.total_trades >= self.ticks_per_bucket:
            self.insert_candle()

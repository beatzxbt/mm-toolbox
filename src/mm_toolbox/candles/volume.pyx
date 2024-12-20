from libc.stdint cimport uint32_t
from .base cimport BaseCandles


class VolumeCandles(BaseCandles):
    def __init__(self, double volume_per_bucket, uint32_t num_candles):
        super().__init__(num_candles)

        self.volume_per_bucket = volume_per_bucket

    cpdef void process_trade(
        self, double time, bint is_buy, double px, double sz 
    ):  
        if self.total_trades == 0.0:
            self.open_time = time
            self.open_price = px

        self.high_price = max(self.high_price, px)
        self.low_price = min(self.low_price, px)
        self.close_price = px

        if is_buy:
            self.buy_volume += sz
        else:
            self.sell_volume += sz

        self.vwap_price = self.calculate_vwap(px, sz)
        self.total_trades += 1.0
        self.close_time = time

        cdef:
            double remaining_volume
            double total_volume = self.buy_volume + self.sell_volume

        if total_volume >= self.volume_per_bucket:
            remaining_volume = total_volume - self.volume_per_bucket

            if is_buy:
                self.buy_volume -= remaining_volume
            else:
                self.sell_volume -= remaining_volume

            self.insert_candle()
            self.process_trade(time, is_buy, px, remaining_volume)

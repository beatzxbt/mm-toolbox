from libc.stdint cimport uint32_t
from .base cimport BaseCandles

cdef class MultiTriggerCandles(BaseCandles):
    def __init__(self, double max_duration_secs, double max_ticks, double max_volume, uint32_t num_candles):
        super().__init__(num_candles)
        
        self.max_duration_millis = max_duration_secs * 1000.0
        self.max_ticks = max_ticks
        self.max_volume = max_volume

    cpdef void process_trade(self, double time, bint is_buy, double px, double sz):
        if self.total_trades == 0.0:
            self.open_time = time
            self.open_price = px

        if self.open_time + self.max_duration_millis <= time:
            self.insert_candle()
            self.process_trade(time, is_buy, px, sz)
            return

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

        if self.total_trades >= self.max_ticks:
            self.insert_candle()
            return

        cdef:
            double remaining_volume
            double total_volume = self.buy_volume + self.sell_volume

        if total_volume > self.max_volume:
            remaining_volume = total_volume - self.max_volume

            if is_buy == 0.0:
                self.buy_volume -= remaining_volume
            else:
                self.sell_volume -= remaining_volume

            self.insert_candle()
            self.process_trade(time, is_buy, px, remaining_volume)
            return

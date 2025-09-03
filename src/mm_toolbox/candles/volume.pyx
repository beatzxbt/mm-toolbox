from libc.math cimport fmax, fmin
from mm_toolbox.candles.base cimport BaseCandles

cdef class VolumeCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on a fixed volume threshold.
    """
    def __init__(self, double volume_per_bucket, int num_candles=1000):
        """Initialize the volume-based candle aggregator."""
        self.volume_per_bucket = volume_per_bucket

    cpdef void process_trade(self, object trade):
        """Process a single trade tick, updating the current candle."""
        cdef:
            double time_ms = trade.time_ms
            bint is_buy = trade.is_buy
            double price = trade.price
            double size = trade.size

        if self.is_stale_trade(time_ms):
            return

        # Initialize a new candle if this is the first trade
        if self.latest_candle.num_trades == 0:
            self.latest_candle.open_time_ms = time_ms
            self.latest_candle.open_price = price

        # Update candle statistics
        self.latest_candle.high_price = fmax(self.latest_candle.high_price, price)
        self.latest_candle.low_price = fmin(self.latest_candle.low_price, price)
        self.latest_candle.close_price = price

        # Update volume based on trade direction
        if is_buy:
            self.latest_candle.buy_size += size
        else:
            self.latest_candle.sell_size += size

        self.latest_candle.vwap_price = self.calculate_vwap(price, size)
        self.latest_candle.trades.append(trade)
        self.latest_candle.num_trades += 1
        self.latest_candle.close_time_ms = time_ms

        cdef:
            double remaining_volume
            double total_volume = self.latest_candle.buy_size + self.latest_candle.sell_size

        # Check if volume threshold has been reached
        if total_volume >= self.volume_per_bucket:
            remaining_volume = total_volume - self.volume_per_bucket

            if is_buy:
                self.latest_candle.buy_size -= remaining_volume
            else:
                self.latest_candle.sell_size -= remaining_volume

            self.insert_and_reset_candle()
            self.process_trade(trade)

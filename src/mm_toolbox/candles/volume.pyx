from libc.math cimport fmax, fmin
from mm_toolbox.candles.base import Trade
from mm_toolbox.candles.base cimport BaseCandles

cdef class VolumeCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on a fixed volume threshold.
    """
    def __init__(self, double volume_per_bucket, int num_candles=1000, bint store_trades=True):
        """Initialize the volume-based candle aggregator."""
        if volume_per_bucket <= 0.0:
            raise ValueError(
                f"Invalid volume_per_bucket; expected >0 but got {volume_per_bucket}"
            )
        BaseCandles.__init__(self, num_candles, store_trades)
        self.volume_per_bucket = volume_per_bucket

    cpdef void process_trade(self, object trade):
        """Process a single trade tick, updating the current candle."""
        cdef:
            double time_ms = trade.time_ms
            bint is_buy = trade.is_buy
            double price = trade.price
            double remaining_size = trade.size
            double total_size
            double available_size
            double chunk_size
            double volume
            double eps = 1e-12

        if self.is_stale_trade(time_ms):
            return

        while remaining_size > eps:
            # Initialize a new candle if this is the first trade
            if self.latest_candle.num_trades == 0:
                self.latest_candle.open_time_ms = time_ms
                self.latest_candle.open_price = price
                self.latest_candle.high_price = price
                self.latest_candle.low_price = price

            total_size = self.latest_candle.buy_size + self.latest_candle.sell_size
            available_size = self.volume_per_bucket - total_size
            if available_size <= eps:
                self.insert_and_reset_candle()
                continue

            chunk_size = remaining_size if remaining_size <= available_size else available_size
            volume = price * chunk_size

            # Update candle statistics
            self.latest_candle.high_price = fmax(self.latest_candle.high_price, price)
            self.latest_candle.low_price = fmin(self.latest_candle.low_price, price)
            self.latest_candle.close_price = price

            # Update volume based on trade direction
            if is_buy:
                self.latest_candle.buy_size += chunk_size
                self.latest_candle.buy_volume += volume
            else:
                self.latest_candle.sell_size += chunk_size
                self.latest_candle.sell_volume += volume

            self.latest_candle.vwap = self.calculate_vwap(price, chunk_size)
            if self._store_trades:
                if chunk_size == remaining_size:
                    self.latest_candle.trades.append(trade)
                else:
                    self.latest_candle.trades.append(
                        Trade(
                            time_ms=trade.time_ms,
                            is_buy=is_buy,
                            price=price,
                            size=chunk_size,
                        )
                    )
            self.latest_candle.num_trades += 1
            self.latest_candle.close_time_ms = time_ms

            remaining_size -= chunk_size
            if remaining_size <= eps:
                remaining_size = 0.0

            if self.latest_candle.buy_size + self.latest_candle.sell_size >= self.volume_per_bucket - eps:
                self.insert_and_reset_candle()

from mm_toolbox.candles.base import Trade
from mm_toolbox.candles.base cimport BaseCandles
from libc.math cimport fmax, fmin

cdef class MultiCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on multiple trigger conditions.
    
    A new candle is created when any of these conditions are met:
    - Maximum duration is reached
    - Maximum number of ticks (trades) is reached
    - Maximum volume is reached
    """
    def __init__(self, double max_duration_secs, int max_ticks, double max_size, int num_candles=1000):
        """Initialize the multi-trigger candle aggregator."""
        if max_duration_secs <= 0.0:
            raise ValueError(
                f"Invalid max_duration_secs; expected >0 but got {max_duration_secs}"
            )
        if max_ticks <= 0:
            raise ValueError(f"Invalid max_ticks; expected >0 but got {max_ticks}")
        if max_size <= 0.0:
            raise ValueError(f"Invalid max_size; expected >0 but got {max_size}")

        BaseCandles.__init__(self, num_candles)
        self.max_duration_millis = max_duration_secs * 1000.0
        self.max_ticks = max_ticks
        self.max_size = max_size

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

            # Check if max duration has been exceeded
            if self.latest_candle.open_time_ms + self.max_duration_millis <= time_ms:
                self.insert_and_reset_candle()
                continue

            # Check if max ticks has already been reached
            if self.latest_candle.num_trades >= self.max_ticks:
                self.insert_and_reset_candle()
                continue

            total_size = self.latest_candle.buy_size + self.latest_candle.sell_size
            available_size = self.max_size - total_size
            if available_size <= eps:
                self.insert_and_reset_candle()
                continue

            chunk_size = remaining_size if remaining_size <= available_size else available_size
            volume = price * chunk_size

            # Update HLC statistics
            self.latest_candle.high_price = fmax(self.latest_candle.high_price, price)
            self.latest_candle.low_price = fmin(self.latest_candle.low_price, price)
            self.latest_candle.close_price = price

            if is_buy:
                self.latest_candle.buy_size += chunk_size
                self.latest_candle.buy_volume += volume
            else:
                self.latest_candle.sell_size += chunk_size
                self.latest_candle.sell_volume += volume

            self.latest_candle.vwap = self.calculate_vwap(price, chunk_size)
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

            if self.latest_candle.num_trades >= self.max_ticks:
                self.insert_and_reset_candle()
                continue

            if self.latest_candle.buy_size + self.latest_candle.sell_size >= self.max_size - eps:
                self.insert_and_reset_candle()

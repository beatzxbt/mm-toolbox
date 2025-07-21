from mm_toolbox.candles.base cimport BaseCandles

cdef class MultiCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on multiple trigger conditions.
    
    A new candle is created when any of these conditions are met:
    - Maximum duration is reached
    - Maximum number of ticks (trades) is reached
    - Maximum volume is reached
    """
    def __init__(self, double max_duration_secs, int max_ticks, double max_sz, int num_candles):
        """
        Initialize the multi-trigger candle aggregator.
        
        Args:
            max_duration_secs (double): Maximum duration of a candle in seconds.
            max_ticks (int): Maximum number of trades in a candle.
            max_sz (double): Maximum size in a candle.
            num_candles (int): Maximum number of candles to store.
        """
        super().__init__(num_candles)
        
        self.max_duration_millis = max_duration_secs * 1000.0
        self.max_ticks = max_ticks
        self.max_sz = max_sz

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz):
        """
        Process a single trade tick, updating the current candle.
        
        Creates a new candle if any trigger condition is met.
        
        Args:
            time_ms (double): The timestamp of the trade in milliseconds.
            is_buy (bint): True if it's a buy trade, False if it's a sell trade.
            px (double): The trade price.
            sz (double): The trade size (volume).
        """ 
        if self.is_stale_trade(time_ms):
            return

        # Initialize a new candle if this is the first trade
        if self.num_trades == 0.0:
            self.open_time_ms = time_ms
            self.open_px = px

        # Check if max duration has been exceeded
        if self.open_time_ms + self.max_duration_millis <= time_ms:
            self.insert_candle()
            self.process_trade(time_ms, is_buy, px, sz)
            return

        # Update candle statistics
        self.high_px = max(self.high_px, px)
        self.low_px = min(self.low_px, px)
        self.close_px = px

        # Update volume based on trade direction
        if is_buy:
            self.buy_sz += sz
        else:
            self.sell_sz += sz

        self.vwap_px = self.calculate_vwap(px, sz)
        self.num_trades += 1.0
        self.close_time_ms = time_ms

        # Check if max ticks has been exceeded
        if self.num_trades >= self.max_ticks:
            self.insert_candle()
            return

        cdef:
            double remaining_sz
            double total_sz = self.buy_sz + self.sell_sz

        if total_sz > self.max_sz:
            remaining_sz = total_sz - self.max_sz

            if is_buy:
                self.buy_sz -= remaining_sz
            else:
                self.sell_sz -= remaining_sz

            self.insert_candle()
            self.process_trade(time_ms, is_buy, px, remaining_sz)
            return

from mm_toolbox.candles.base cimport BaseCandles

cdef class TickCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on a fixed number of trades.
    
    A new candle is created when the specified number of trades is reached.
    """
    def __init__(self, int ticks_per_bucket, int num_candles):
        """
        Initialize the tick-based candle aggregator.
        
        Args:
            ticks_per_bucket (int): Number of trades per candle.
            num_candles (int): Maximum number of candles to store.
        """
        super().__init__(num_candles)
        
        self.ticks_per_bucket = ticks_per_bucket

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz):
        """
        Process a single trade tick, updating the current candle.
        
        Creates a new candle when the number of trades reaches ticks_per_bucket.
        
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
            self.high_px = px
            self.low_px = px

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

        # Check if max ticks has been reached
        if self.num_trades >= self.ticks_per_bucket:
            self.insert_candle()

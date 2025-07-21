from mm_toolbox.candles.base cimport BaseCandles

cdef class PriceCandles(BaseCandles):
    """
    Candle aggregator that creates new candles based on price movement.
    
    A new candle is created when the price moves by a specified amount from the opening price.
    """
    def __init__(self, double price_bucket, int num_candles):
        """
        Initialize the price-based candle aggregator.
        
        Args:
            price_bucket (double): Price movement threshold that triggers a new candle.
            num_candles (int): Maximum number of candles to store.
        """
        super().__init__(num_candles)
        
        self.px_bucket = price_bucket
        self.upper_px_bound = 0.0
        self.lower_px_bound = 0.0

    cpdef void process_trade(self, double time_ms, bint is_buy, double px, double sz):
        """
        Process a single trade tick, updating the current candle.
        
        Creates a new candle when the price moves by price_bucket amount from the opening price.
        
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

            # Set price bounds
            self.upper_px_bound = px + self.px_bucket
            self.lower_px_bound = px - self.px_bucket

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

        # Check if price movement threshold has been reached
        if px > self.upper_px_bound or px < self.lower_px_bound:
            self.insert_candle()

cdef class DataBoundsFilter:
    """Filters data based on a threshold percentage."""

    def __cinit__(self, double threshold_pct):
        self._threshold = threshold_pct / 100.0
        self._lower_bound = 0.0
        self._upper_bound = 0.0

    cpdef bint check_and_update(self, double value, bint reset=False):
        """Checks if value is outside current bounds or if reset is True, updates bounds if so."""
        cdef double lower, upper, threshold
        threshold = self._threshold

        if reset:
            self._lower_bound = value * (1.0 - threshold)
            self._upper_bound = value * (1.0 + threshold)
            return True

        lower = self._lower_bound
        upper = self._upper_bound

        if value < lower or value > upper:
            self._lower_bound = value * (1.0 - threshold)
            self._upper_bound = value * (1.0 + threshold)
            return True

        return False

def _example() -> None:
    """Example of filtering mock Binance BBO price/size data."""
    size_filter = DataBoundsFilter(threshold_pct=10.0)

    # Simulated Binance BBO stream: (price, size)
    bbo_data = [
        (10000.0, 0.5),
        (10001.0, 0.52),
        (10010.0, 0.51), 
        (10009.0, 0.50),
        (10009.5, 0.80),   # size jump, should trigger filter
        (10009.6, 0.81),
        (10009.7, 0.79),
    ]
    
    previous_price = 0.0
    for price, size in bbo_data:
        price_changed = price != previous_price
        if price_changed:
            price_pass = size_filter.check_and_update(size, reset=True)
            previous_price = price
        else:
            price_pass = True

        size_pass = size_filter.check_and_update(size)
        print(f"Price: {price}, Size: {size} | PricePass: {price_pass}, SizePass: {size_pass}")
    
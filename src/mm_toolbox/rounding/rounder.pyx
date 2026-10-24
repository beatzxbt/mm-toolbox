"""
Rounding utilities for prices and sizes in financial markets.

Provides configurable rounding of prices and sizes to tick and lot size
multiples. Supports both scalar and NumPy array operations with
bid/ask/size-specific rounding directions.
"""
import msgspec
import math
import numpy as np
cimport numpy as cnp
from numpy cimport (
    ndarray as cndarray,
    PyArray_EMPTY as cPyArray_EMPTY,
)
from libc.math cimport ceil, floor, fabs, isfinite


def _power10_exponent(double value, str field):
    """Validate canonical finite decimal powers once, outside rounding paths."""
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid {field}; must be greater than 0 and finite")
    exponent = int(round(math.log10(value)))
    if exponent < -308 or exponent > 308 or value != float(f"1e{exponent}"):
        raise ValueError(f"Invalid {field}; must be a power of ten in [1e-308, 1e308]")
    return exponent

class RounderConfig(msgspec.Struct):
    tick_size: float
    lot_size: float
    round_bids_down: bool
    round_asks_up: bool
    round_size_up: bool

    def __post_init__(self) -> None:
        _power10_exponent(self.tick_size, "tick_size")
        _power10_exponent(self.lot_size, "lot_size")

    @classmethod
    def default(cls, tick_size: float, lot_size: float) -> "RounderConfig":
        return cls(
            tick_size=tick_size,
            lot_size=lot_size,
            round_bids_down=True,
            round_asks_up=True,
            round_size_up=True,
        )

cdef class Rounder:
    """Provides rounding operations on prices and sizes according to specified tick and lot sizes.

    Attributes:
        config: RounderConfig instance with tick/lot sizes and rounding directions.
        tick_size: Minimum price increment.
        lot_size: Minimum size increment.
    """

    def __cinit__(self, object config):
        """Initialize a new Rounder.

        Args:
            config: RounderConfig with tick_size, lot_size, and rounding flags.
        """
        self.config: RounderConfig = config
        self.tick_size = config.tick_size
        self.lot_size = config.lot_size

        self._bid_up = not config.round_bids_down
        self._ask_up = config.round_asks_up
        self._size_up = config.round_size_up

        # Derive reciprocal scales from decimal exponents, avoiding 1 / tick drift.
        tick_exp = _power10_exponent(self.tick_size, "tick_size")
        lot_exp = _power10_exponent(self.lot_size, "lot_size")
        self._inverse_tick_size = float(f"1e{-tick_exp}")
        self._inverse_lot_size = float(f"1e{-lot_exp}")
        self._tick_rounding_factor = float(f"1e{abs(tick_exp)}")
        self._lot_rounding_factor = float(f"1e{abs(lot_exp)}")

    cdef inline double _grid_value(self, double ticks, double step, double factor) noexcept nogil:
        """Reconstruct the nearest float to a decimal grid point."""
        return ticks / factor if step < 1.0 else ticks * step

    cdef inline double _quantize(self, double value, double step, double inverse, double factor, bint up):
        """Round on a decimal grid, correcting one-tick quotient boundary errors."""
        cdef double quotient = value * inverse
        cdef double ticks, result, adjacent
        if not isfinite(value) or not isfinite(quotient) or fabs(quotient) > 4503599627370495.0:
            raise ValueError("Value must be finite and within the supported 2**52 tick range")
        ticks = ceil(quotient) if up else floor(quotient)
        result = self._grid_value(ticks, step, factor)
        if up:
            if result < value:
                ticks += 1.0
            else:
                adjacent = self._grid_value(ticks - 1.0, step, factor)
                if adjacent >= value:
                    ticks -= 1.0
        else:
            if result > value:
                ticks -= 1.0
            else:
                adjacent = self._grid_value(ticks + 1.0, step, factor)
                if adjacent <= value:
                    ticks += 1.0
        result = self._grid_value(ticks, step, factor)
        if not isfinite(result):
            raise ValueError("Rounded result exceeds finite float range")
        return result

    cpdef double bid(self, double price):
        """Round a bid price to the nearest tick size multiple.

        Args:
            price: Price to round.

        Returns:
            Rounded bid price.
        """
        return self._quantize(price, self.tick_size, self._inverse_tick_size, self._tick_rounding_factor, self._bid_up)

    cpdef double ask(self, double price):
        """Round an ask price to the nearest tick size multiple.

        Args:
            price: Price to round.

        Returns:
            Rounded ask price.
        """
        return self._quantize(price, self.tick_size, self._inverse_tick_size, self._tick_rounding_factor, self._ask_up)

    cpdef double size(self, double size):
        """Round a size to the nearest lot size multiple.

        Args:
            size: Size to round.

        Returns:
            Rounded size.
        """
        return self._quantize(size, self.lot_size, self._inverse_lot_size, self._lot_rounding_factor, self._size_up)

    cpdef cnp.ndarray bids(self, cnp.ndarray prices):
        """Round an array of prices to the nearest tick size multiple (bids).

        Args:
            prices: NumPy array of prices.

        Returns:
            NumPy array of rounded bid prices.
        """
        cdef:
            double[:] prices_view = prices
            Py_ssize_t i, n = prices_view.shape[0]
            cnp.npy_intp dim = n
            cndarray result = <cndarray> cPyArray_EMPTY(1, &dim, cnp.NPY_FLOAT64, 0)
            double[:] result_view = result
            
            # Fast local copies of instance variables for faster access in the loop
            double tick_size = self.tick_size
            double inverse_tick_size = self._inverse_tick_size
            double tick_rounding_factor = self._tick_rounding_factor

        for i in range(n):
            result_view[i] = self._quantize(
                prices_view[i], tick_size, inverse_tick_size,
                tick_rounding_factor, self._bid_up
            )
        return result

    cpdef cnp.ndarray asks(self, cnp.ndarray prices):
        """Round an array of prices to the nearest tick size multiple (asks).

        Args:
            prices: NumPy array of prices.

        Returns:
            NumPy array of rounded ask prices.
        """
        cdef:
            double[:] prices_view = prices
            Py_ssize_t i, n = prices_view.shape[0]
            cnp.npy_intp dim = n
            cndarray result = <cndarray> cPyArray_EMPTY(1, &dim, cnp.NPY_FLOAT64, 0)
            double[:] result_view = result

            # Fast local copies of instance variables for faster access in the loop
            double tick_size = self.tick_size
            double inverse_tick_size = self._inverse_tick_size
            double tick_rounding_factor = self._tick_rounding_factor

        for i in range(n):
            result_view[i] = self._quantize(
                prices_view[i], tick_size, inverse_tick_size,
                tick_rounding_factor, self._ask_up
            )
        return result

    cpdef cnp.ndarray sizes(self, cnp.ndarray sizes):
        """Round an array of sizes to the nearest lot size multiple.

        Args:
            sizes: NumPy array of sizes.

        Returns:
            NumPy array of rounded sizes.
        """
        cdef:
            double[:] sizes_view = sizes
            Py_ssize_t i, n = sizes_view.shape[0]
            cnp.npy_intp dim = n
            cndarray result = <cndarray> cPyArray_EMPTY(1, &dim, cnp.NPY_FLOAT64, 0)
            double[:] result_view = result

            # Fast local copies of instance variables for faster access in the loop
            double lot_size = self.lot_size
            double inverse_lot_size = self._inverse_lot_size
            double lot_rounding_factor = self._lot_rounding_factor

        for i in range(n):
            result_view[i] = self._quantize(
                sizes_view[i], lot_size, inverse_lot_size,
                lot_rounding_factor, self._size_up
            )
        return result

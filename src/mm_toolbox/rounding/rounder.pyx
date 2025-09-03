import msgspec
import numpy as np
cimport numpy as cnp
from numpy cimport (
    ndarray as cndarray,
    PyArray_EMPTY as cPyArray_EMPTY,
)
from libc.math cimport ceil, floor, log10, pow, round as c_round

class RounderConfig(msgspec.Struct):
    tick_size: float
    lot_size: float
    round_bids_down: bool
    round_asks_up: bool
    round_size_up: bool

    def __post_init__(self) -> None:
        if self.tick_size <= 0.0:
            raise ValueError("Invalid tick_size; must be greater than 0")
        if self.lot_size <= 0.0:
            raise ValueError("Invalid lot_size; must be greater than 0")

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
    """Provides rounding operations on prices and sizes according to specified tick and lot sizes."""

    def __cinit__(self, object config):
        self.config: RounderConfig = config
        self.tick_size = config.tick_size
        self.lot_size = config.lot_size

        self._bid_round_func = floor if config.round_bids_down else ceil
        self._ask_round_func = ceil if config.round_asks_up else floor
        self._size_round_func = ceil if config.round_size_up else floor

        # Precompute frequently used values
        self._inverse_tick_size = 1.0 / self.tick_size
        self._inverse_lot_size = 1.0 / self.lot_size
        self._tick_rounding_factor = pow(10.0, ceil(-log10(self.tick_size)))
        self._lot_rounding_factor = pow(10.0, ceil(-log10(self.lot_size)))

    cdef inline double _round_to(self, double num, double factor) nogil:
        """Round a value to a specific decimal precision."""
        return c_round(num * factor) / factor

    cpdef double bid(self, double price):
        """Round a price to the nearest tick size multiple, direction depends on config."""
        cdef double value = self.tick_size * self._bid_round_func(price * self._inverse_tick_size)
        return self._round_to(value, self._tick_rounding_factor)

    cpdef double ask(self, double price):
        """Round a price up to the nearest tick size multiple."""
        cdef double value = self.tick_size * self._ask_round_func(price * self._inverse_tick_size)
        return self._round_to(value, self._tick_rounding_factor)

    cpdef double size(self, double size):
        """Round a size down to the nearest lot size multiple."""
        cdef double value = self.lot_size * self._size_round_func(size * self._inverse_lot_size)
        return self._round_to(value, self._lot_rounding_factor)

    cpdef cnp.ndarray bids(self, cnp.ndarray prices):
        """Round an array of prices down to the nearest tick size multiple (bids)."""
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
            result_view[i] = self._round_to(
                tick_size * self._bid_round_func(prices_view[i] * inverse_tick_size),
                tick_rounding_factor
            )
        return result

    cpdef cnp.ndarray asks(self, cnp.ndarray prices):
        """Round an array of prices up to the nearest tick size multiple (asks)."""
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
            result_view[i] = self._round_to(
                tick_size * self._ask_round_func(prices_view[i] * inverse_tick_size),
                tick_rounding_factor
            )
        return result

    cpdef cnp.ndarray sizes(self, cnp.ndarray sizes):
        """Round an array of sizes down to the nearest lot size multiple."""
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
            result_view[i] = self._round_to(
                lot_size * self._size_round_func(sizes_view[i] * inverse_lot_size),
                lot_rounding_factor
            )
        return result

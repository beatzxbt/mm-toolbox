# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
OrderbookLevel data structures and construction utilities.

Provides C-level factory functions for creating OrderbookLevel structs and
Python-facing wrapper classes (PyOrderbookLevel, PyOrderbookLevels) for level
management. Includes multiple input formats: lists, numpy arrays, with optional
pre-computed ticks/lots for performance optimization.
"""
from __future__ import annotations

from typing import Self

from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from .level cimport OrderbookLevel, OrderbookLevels

from .helpers cimport (
    convert_price_to_tick,
    convert_size_to_lot,
)

cdef inline OrderbookLevel create_orderbook_level(double price, double size, u64 norders=1) noexcept nogil:
    """Create an OrderbookLevel struct without pre-computing ticks/lots."""
    cdef OrderbookLevel level = OrderbookLevel()
    level.price = price
    level.size = size
    level.norders = norders
    level.ticks = 0
    level.lots = 0
    level.__padding1 = 0
    level.__padding2 = 0
    level.__padding3 = 0
    return level

cdef inline OrderbookLevel create_orderbook_level_with_ticks_and_lots(
    double price,
    double size,
    double tick_size,
    double lot_size,
    u64 norders=1,
) noexcept nogil:
    """Create an OrderbookLevel struct with pre-computed ticks and lots."""
    cdef OrderbookLevel level = OrderbookLevel()
    level.price = price
    level.size = size
    level.norders = norders
    level.ticks = convert_price_to_tick(price, tick_size)
    level.lots = convert_size_to_lot(size, lot_size)
    level.__padding1 = 0
    level.__padding2 = 0
    level.__padding3 = 0
    return level

cdef inline OrderbookLevels create_orderbook_levels(u64 num_levels, OrderbookLevel* levels) noexcept nogil:
    """Create an OrderbookLevels struct from a pointer and count."""
    cdef OrderbookLevels levels_struct = OrderbookLevels()
    levels_struct.num_levels = num_levels
    levels_struct.levels = levels
    return levels_struct


cdef inline void free_orderbook_levels(OrderbookLevels* levels) noexcept nogil:
    """Free the memory allocated for OrderbookLevels.levels if not NULL."""
    if levels != NULL and levels.levels != NULL:
        free(<void*> levels.levels)
        levels.levels = NULL
        levels.num_levels = 0


cdef class PyOrderbookLevel:
    """Python wrapper around the C OrderbookLevel struct."""
    def __cinit__(
        self, 
        double price, 
        double size, 
        u64 norders=1, 
        u64 ticks=0, 
        u64 lots=0, 
        bint verify_values=True
    ) -> None:
        """
        Create a new PyOrderbookLevel instance.

        Args:
            price: The price of the orderbook level.
            size: The size of the orderbook level.
            norders: The number of orders at the orderbook level.
            ticks: The ticks of the orderbook level.
            lots: The lots of the orderbook level.
            verify_values: Whether to verify the values of the orderbook level.
        """
        if verify_values:
            if price <= 0.0:
                raise ValueError(f"Invalid price; expected >0 but got {price}")
            if size < 0.0:
                raise ValueError(f"Invalid size; expected >=0 but got {size}")
            if norders < 0:
                raise ValueError(f"Invalid norders; expected >=0 but got {norders}")
            if ticks < 0:
                raise ValueError(f"Invalid ticks; expected >=0 but got {ticks}")
            if lots < 0:
                raise ValueError(f"Invalid lots; expected >=0 but got {lots}")

        # Create the struct directly (can't use create_orderbook_level since it doesn't accept ticks/lots)
        self._level.price = price
        self._level.size = size
        self._level.norders = norders
        self._level.ticks = ticks
        self._level.lots = lots
        self._level.__padding1 = 0
        self._level.__padding2 = 0
        self._level.__padding3 = 0

    @staticmethod
    def with_ticks_and_lots(
        double price, 
        double size, 
        double tick_size, 
        double lot_size, 
        u64 norders=1, 
        bint verify_values=True
    ) -> Self:
        """Create a new PyOrderbookLevel instance with ticks and lots."""
        cdef PyOrderbookLevel wrapper = PyOrderbookLevel(
            price=price,
            size=size,
            norders=norders,
            ticks=convert_price_to_tick(price, tick_size),
            lots=convert_size_to_lot(size, lot_size),
            verify_values=verify_values,
        )
        return wrapper

    @staticmethod
    cdef PyOrderbookLevel from_struct(OrderbookLevel level):
        """Create a PyOrderbookLevel from a C OrderbookLevel struct.
        
        Args:
            level: The C OrderbookLevel struct to wrap.
            
        Returns:
            A new PyOrderbookLevel instance wrapping the struct.
        """
        cdef PyOrderbookLevel wrapper = PyOrderbookLevel.__new__(PyOrderbookLevel)
        wrapper._level = level
        return wrapper

    cdef OrderbookLevel to_c_struct(self):
        """Return the underlying C OrderbookLevel struct (Cython only)."""
        return self._level

    @property
    def price(self) -> float:
        """Get the price of this level."""
        return self._level.price

    @property
    def size(self) -> float:
        """Get the size of this level."""
        return self._level.size

    @property
    def norders(self) -> int:
        """Get the number of orders at this level."""
        return self._level.norders

    @property
    def ticks(self) -> int:
        """Get the price in ticks."""
        return self._level.ticks

    @property
    def lots(self) -> int:
        """Get the size in lots."""
        return self._level.lots

    def __repr__(self) -> str:
        """Return a print-friendly representation of the PyOrderbookLevel instance."""
        return f"PyOrderbookLevel(price={self._level.price}, size={self._level.size}, norders={self._level.norders}, ticks={self._level.ticks}, lots={self._level.lots})"


cdef class PyOrderbookLevels:
    """Python wrapper around the C OrderbookLevels struct."""
    
    @staticmethod
    cdef PyOrderbookLevels _create(u64 num_levels, OrderbookLevel* levels):
        """Internal factory for creating PyOrderbookLevels from C types."""
        cdef PyOrderbookLevels obj = PyOrderbookLevels.__new__(PyOrderbookLevels)
        obj._levels = create_orderbook_levels(num_levels, levels)
        return obj
    
    def __cinit__(self) -> None:
        """Initialize an empty PyOrderbookLevels instance."""
        # Fields initialized by _create or factory methods
        self._levels.num_levels = 0
        self._levels.levels = NULL

    def __dealloc__(self) -> None:
        """Deallocate the PyOrderbookLevels instance."""
        free_orderbook_levels(&self._levels)
        # Note: _levels is a struct, free_orderbook_levels already resets fields

    @staticmethod
    cdef PyOrderbookLevels from_ptr(OrderbookLevel* levels_ptr, u64 num_levels):
        """Create a new PyOrderbookLevels instance from a pointer and count (Cython only)."""
        if levels_ptr == NULL:
            raise ValueError("Invalid levels_ptr; expected non-null")
        if num_levels <= 0:
            raise ValueError(f"Invalid num_levels; expected >0 but got {num_levels}")

        cdef u64 size = sizeof(OrderbookLevel) * num_levels
        cdef OrderbookLevel* new_levels = <OrderbookLevel*> malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        memcpy(new_levels, levels_ptr, size)

        return PyOrderbookLevels._create(num_levels, new_levels)

    @staticmethod
    def from_list(list[double] prices, list[double] sizes, list[u64] norders=None, bint verify_values=True) -> Self:
        """Create a new PyOrderbookLevels instance from a list of prices, sizes, and norders."""
        if verify_values:
            len_prices = len(prices)
            len_sizes = len(sizes)
            len_norders = len(norders) if norders is not None else 0
            if len_prices <= 0:
                raise ValueError(f"Invalid prices; expected >0 but got {len_prices}")
            if len_sizes != len_prices:
                raise ValueError(f"Mismatched lengths; expected same length for prices and sizes but got {len_prices} and {len_sizes}")
            if norders is not None and len_norders != len_prices:
                raise ValueError(f"Mismatched lengths; expected same length for prices and norders but got {len_prices} and {len_norders}")

        cdef:
            u64 i
            u64 size = sizeof(OrderbookLevel) * len_prices
            bint use_default = norders is None or len_norders == 0
            OrderbookLevel* new_levels = <OrderbookLevel*> malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        try:
            for i in range(len_prices):
                new_levels[i] = create_orderbook_level(
                    price=prices[i],
                    size=sizes[i],
                    norders=norders[i] if not use_default else 1,
                )
        except:
            free(new_levels)
            raise

        return PyOrderbookLevels._create(<u64>len_prices, new_levels)

    @staticmethod
    def from_list_with_ticks_and_lots(
        list[double] prices,
        list[double] sizes,
        list[u64] norders,
        double tick_size,
        double lot_size,
        bint verify_values=True,
    ) -> Self:
        """Create a new PyOrderbookLevels instance from lists with pre-computed ticks and lots."""
        cdef Py_ssize_t num = len(prices)
        if verify_values:
            if num <= 0:
                raise ValueError(f"Invalid prices; expected >0 but got {num}")
            if len(sizes) != num:
                raise ValueError(f"Mismatched lengths; expected same length for prices and sizes but got {num} and {len(sizes)}")
            if len(norders) != num:
                raise ValueError(f"Mismatched lengths; expected same length for prices and norders but got {num} and {len(norders)}")

        cdef:
            u64 i
            u64 size = sizeof(OrderbookLevel) * num
            OrderbookLevel* new_levels = <OrderbookLevel*> malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        try:
            for i in range(num):
                new_levels[i] = create_orderbook_level_with_ticks_and_lots(
                    prices[i],
                    sizes[i],
                    tick_size,
                    lot_size,
                    norders[i],
                )
        except:
            free(new_levels)
            raise

        return PyOrderbookLevels._create(<u64>num, new_levels)

    @staticmethod
    def from_numpy(double[:] prices, double[:] sizes, u64[:] norders=None, bint verify_values=True) -> Self:
        """Create a new PyOrderbookLevels instance from numpy arrays."""
        cdef Py_ssize_t num = prices.shape[0]
        cdef bint use_default = norders is None

        if verify_values:
            if num <= 0:
                raise ValueError(f"Invalid prices; expected >0 but got {num}")
            if sizes.shape[0] != num:
                raise ValueError(f"Mismatched lengths; expected same length for prices and sizes but got {num} and {sizes.shape[0]}")
            if not use_default and norders.shape[0] != num:
                raise ValueError(f"Mismatched lengths; expected same length for prices and norders but got {num} and {norders.shape[0]}")

        cdef:
            Py_ssize_t i
            u64 size = sizeof(OrderbookLevel) * num
            OrderbookLevel* new_levels = <OrderbookLevel*> malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        try:
            for i in range(num):
                new_levels[i] = create_orderbook_level(
                    price=prices[i],
                    size=sizes[i],
                    norders=norders[i] if not use_default else 1,
                )
        except:
            free(new_levels)
            raise

        return PyOrderbookLevels._create(<u64>num, new_levels)

    @staticmethod
    def from_numpy_with_ticks_and_lots(
        double[:] prices,
        double[:] sizes,
        u64[:] norders,
        double tick_size,
        double lot_size,
        bint verify_values=True,
    ) -> Self:
        """Create a new PyOrderbookLevels instance from numpy arrays with pre-computed ticks and lots."""
        cdef Py_ssize_t num = prices.shape[0]

        if verify_values:
            if num <= 0:
                raise ValueError(f"Invalid prices; expected >0 but got {num}")
            if sizes.shape[0] != num:
                raise ValueError(f"Mismatched lengths; expected same length for prices and sizes but got {num} and {sizes.shape[0]}")
            if norders.shape[0] != num:
                raise ValueError(f"Mismatched lengths; expected same length for prices and norders but got {num} and {norders.shape[0]}")

        cdef:
            Py_ssize_t i
            u64 size = sizeof(OrderbookLevel) * num
            OrderbookLevel* new_levels = <OrderbookLevel*> malloc(size)

        if new_levels == NULL:
            raise MemoryError("Failed to allocate memory for new levels")

        try:
            for i in range(num):
                new_levels[i] = create_orderbook_level_with_ticks_and_lots(
                    prices[i],
                    sizes[i],
                    tick_size,
                    lot_size,
                    norders[i],
                )
        except:
            free(new_levels)
            raise

        return PyOrderbookLevels._create(<u64>num, new_levels)

    cdef OrderbookLevels to_c_struct(self):
        """Return the underlying C OrderbookLevels struct."""
        return self._levels

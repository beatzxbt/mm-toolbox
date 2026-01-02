# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c
# distutils: sources = src/mm_toolbox/orderbook/advanced/c/orderbook_ladder.c
# distutils: include_dirs = src/mm_toolbox/orderbook/advanced/c

"""
Orderbook ladder management for bid/ask level arrays.

This module provides the OrderbookLadder class which manages a single side
(bids or asks) of an orderbook with fixed-size level storage. Performance-
critical operations (roll_right, roll_left, insert_level) delegate to C
implementations for optimal memory movement.
"""
from __future__ import annotations

import numpy as np
cimport numpy as cnp
from cpython.ref cimport Py_INCREF
from libc.stdint cimport uint64_t as u64
from libc.stdlib cimport free, malloc
from posix.stdlib cimport posix_memalign

from ..level.level cimport OrderbookLevel

# Initialize NumPy C API
cnp.import_array()

# Extern declaration for setting array base object
cdef extern from "numpy/arrayobject.h":
    int PyArray_SetBaseObject(cnp.ndarray arr, object obj)


# C declarations from orderbook headers
cdef extern from "orderbook_types.h":
    u64 ORDERBOOK_MAX_LEVELS

cdef extern from "orderbook_ladder.h":
    ctypedef struct OrderbookLadderData:
        u64 num_levels
        u64 max_levels
        OrderbookLevel* levels
        int is_price_ascending

    void c_ladder_roll_right(OrderbookLadderData* data, u64 start_index) nogil
    void c_ladder_roll_left(OrderbookLadderData* data, u64 start_index) nogil
    void c_ladder_insert_level(OrderbookLevel* levels, u64 index, const OrderbookLevel* level) nogil


cdef class OrderbookLadder:
    """Manages a single ladder (bids or asks) with fixed-size level storage."""

    def __cinit__(self, u64 max_levels, bint is_price_ascending) -> None:
        if max_levels == 0:
            raise ValueError(f"Invalid max_levels; expected >0 but got {max_levels}")
        if max_levels > ORDERBOOK_MAX_LEVELS:
            raise ValueError(
                f"Invalid max_levels; expected <={ORDERBOOK_MAX_LEVELS} but got {max_levels} "
                f"(prevents integer overflow in memory allocation)"
            )

        cdef void* raw_ptr
        cdef int mem_align_errno = posix_memalign(&raw_ptr, 64, max_levels * sizeof(OrderbookLevel))
        if mem_align_errno != 0:
            raw_ptr = malloc(max_levels * sizeof(OrderbookLevel))
            if raw_ptr == NULL:
                raise MemoryError(f"Cannot allocate memory for levels; posix_memalign errno: {mem_align_errno}")
        self._levels = <OrderbookLevel*> raw_ptr

        levels_dtype = np.dtype(
            [
                ("price", np.double),
                ("size", np.double),
                ("norders", np.uint64),
                ("ticks", np.uint64),
                ("lots", np.uint64),
                ("__padding1", np.uint64),
                ("__padding2", np.uint64),
                ("__padding3", np.uint64),
            ],
            align=True,
        )
        # Create numpy array from raw memory using PyArray_SimpleNewFromData
        # This avoids Cython memoryview creating its own base reference
        cdef cnp.npy_intp dims[1]
        dims[0] = <cnp.npy_intp>(max_levels * sizeof(OrderbookLevel))
        cdef cnp.ndarray base_array = cnp.PyArray_SimpleNewFromData(
            1, dims, cnp.NPY_UINT8,
            <void*>self._levels
        )
        # Set self as the base so that views keep self alive
        Py_INCREF(self)  # PyArray_SetBaseObject steals a reference
        PyArray_SetBaseObject(base_array, self)
        # Create the structured view with proper dtype
        self._levels_numpy = base_array.view(dtype=levels_dtype)

        self._data = OrderbookLadderData()
        self._data.num_levels = 0
        self._data.max_levels = max_levels
        self._data.levels = self._levels
        self._data.is_price_ascending = is_price_ascending

    def __dealloc__(self):
        if self._levels != NULL:
            free(<void*> self._levels)
        self._levels = NULL

    cdef inline OrderbookLadderData* get_data(self) noexcept nogil:
        """Return a pointer to the ladder data for read-only access."""
        return &self._data

    cdef void insert_level(self, u64 index, OrderbookLevel level) noexcept nogil:
        """Insert a level at the specified index."""
        c_ladder_insert_level(self._levels, index, &level)

    cdef void roll_right(self, u64 start_index) noexcept nogil:
        """Shift levels right starting from start_index to make room for insertion.

        Note: This only shifts data, it does NOT update the count. Caller must
        call increment_count() separately if a new level is being added.
        """
        c_ladder_roll_right(&self._data, start_index)

    cdef void roll_left(self, u64 start_index) noexcept nogil:
        """Shift levels left starting from start_index to remove a level.

        Note: This only shifts data, it does NOT update the count. Caller must
        call decrement_count() separately after removing a level.
        """
        c_ladder_roll_left(&self._data, start_index)

    cdef inline void reset(self) noexcept nogil:
        """Reset the ladder to empty state."""
        self._data.num_levels = 0

    cdef inline void increment_count(self) noexcept nogil:
        """Increment the level count if not at max capacity."""
        if self._data.num_levels < self._data.max_levels:
            self._data.num_levels += 1

    cdef inline void decrement_count(self) noexcept nogil:
        """Decrement the level count if not empty."""
        if self._data.num_levels > 0:
            self._data.num_levels -= 1

    cdef inline bint is_empty(self) noexcept nogil:
        """Check if the ladder has no levels."""
        return self._data.num_levels == 0

    cdef inline bint is_full(self) noexcept nogil:
        """Check if the ladder is at max capacity."""
        return self._data.num_levels == self._data.max_levels

    cpdef get_levels(self, bint copy=False):
        """Return a NumPy array of all levels.

        Args:
            copy: If True, return a copy of the data. If False (default), return a view.
                  WARNING: Views share memory with this OrderbookLadder. You must keep
                  the OrderbookLadder object alive for as long as you use the view,
                  otherwise you'll access freed memory.

        Returns:
            NumPy structured array with fields: price, size, norders, ticks, lots
        """
        cdef object result = self._levels_numpy[: self._data.num_levels]
        if copy:
            return result.copy()
        return result

    cpdef get_prices(self, bint copy=False):
        """Return a NumPy array of prices.

        Args:
            copy: If True, return a copy. If False, return a view (see get_levels() for warnings).
        """
        cdef object result = self._levels_numpy["price"][: self._data.num_levels]
        if copy:
            return result.copy()
        return result

    cpdef get_sizes(self, bint copy=False):
        """Return a NumPy array of sizes.

        Args:
            copy: If True, return a copy. If False, return a view (see get_levels() for warnings).
        """
        cdef object result = self._levels_numpy["size"][: self._data.num_levels]
        if copy:
            return result.copy()
        return result

    cpdef get_norders(self, bint copy=False):
        """Return a NumPy array of norders.

        Args:
            copy: If True, return a copy. If False, return a view (see get_levels() for warnings).
        """
        cdef object result = self._levels_numpy["norders"][: self._data.num_levels]
        if copy:
            return result.copy()
        return result

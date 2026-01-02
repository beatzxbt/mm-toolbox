"""Type stubs for level.pyx - C-level orderbook level structures and utilities."""

from __future__ import annotations

from typing import Any, List, Optional, Self

import numpy as np
import numpy.typing as npt

class OrderbookLevel:
    """C struct representing a single orderbook level.

    Note: This is a C struct and is primarily used internally via cimport.
    """

    price: float
    size: float
    norders: int
    ticks: int
    lots: int

class OrderbookLevels:
    """C struct representing a collection of orderbook levels.

    Note: This is a C struct and is primarily used internally via cimport.
    """

    num_levels: int
    levels: Any  # OrderbookLevel* pointer

class PyOrderbookLevel:
    """Python wrapper around the C OrderbookLevel struct."""

    @property
    def price(self) -> float:
        """Get the price of this level."""
        ...

    @property
    def size(self) -> float:
        """Get the size of this level."""
        ...

    @property
    def norders(self) -> int:
        """Get the number of orders at this level."""
        ...

    @property
    def ticks(self) -> int:
        """Get the price in ticks."""
        ...

    @property
    def lots(self) -> int:
        """Get the size in lots."""
        ...

    def __init__(
        self,
        price: float,
        size: float,
        norders: int = 1,
        ticks: int = 0,
        lots: int = 0,
        verify_values: bool = True,
    ) -> None:
        """Create a new PyOrderbookLevel instance.

        Args:
            price: The price of the orderbook level (must be > 0)
            size: The size of the orderbook level (must be >= 0)
            norders: The number of orders at the orderbook level (must be >= 0)
            ticks: The price in ticks (must be >= 0)
            lots: The size in lots (must be >= 0)
            verify_values: Whether to verify the values of the orderbook level

        Raises:
            ValueError: If verify_values is True and any value is invalid
        """
        ...

    @staticmethod
    def with_ticks_and_lots(
        price: float,
        size: float,
        tick_size: float,
        lot_size: float,
        norders: int = 1,
        verify_values: bool = True,
    ) -> Self:
        """Create a new PyOrderbookLevel instance with pre-computed ticks and lots.

        Args:
            price: The price of the orderbook level
            size: The size of the orderbook level
            tick_size: The tick size for computing ticks
            lot_size: The lot size for computing lots
            norders: The number of orders at the orderbook level
            verify_values: Whether to verify the values

        Returns:
            A new PyOrderbookLevel instance with ticks and lots computed
        """
        ...

    def __repr__(self) -> str:
        """Return a print-friendly representation of the PyOrderbookLevel instance."""
        ...

    # Note: The following methods are cdef-only:
    # @staticmethod
    # cdef PyOrderbookLevel from_struct(OrderbookLevel level)
    # cdef OrderbookLevel to_c_struct(self)

class PyOrderbookLevels:
    """Python wrapper around the C OrderbookLevels struct."""

    def __init__(self) -> None:
        """Initialize an empty PyOrderbookLevels instance.

        Note: Use the factory methods (from_list, from_numpy, etc.) to create
        populated instances.
        """
        ...

    @staticmethod
    def from_list(
        prices: List[float],
        sizes: List[float],
        norders: Optional[List[int]] = None,
        verify_values: bool = True,
    ) -> Self:
        """Create a new PyOrderbookLevels instance from lists.

        Args:
            prices: List of prices
            sizes: List of sizes (must match length of prices)
            norders: Optional list of norders (must match length of prices if provided)
            verify_values: Whether to verify the values

        Returns:
            A new PyOrderbookLevels instance

        Raises:
            ValueError: If verify_values is True and lists have invalid lengths
            MemoryError: If memory allocation fails
        """
        ...

    @staticmethod
    def from_list_with_ticks_and_lots(
        prices: List[float],
        sizes: List[float],
        norders: List[int],
        tick_size: float,
        lot_size: float,
        verify_values: bool = True,
    ) -> Self:
        """Create a new PyOrderbookLevels instance from lists with pre-computed ticks and lots.

        Args:
            prices: List of prices
            sizes: List of sizes (must match length of prices)
            norders: List of norders (must match length of prices)
            tick_size: The tick size for computing ticks
            lot_size: The lot size for computing lots
            verify_values: Whether to verify the values

        Returns:
            A new PyOrderbookLevels instance with ticks and lots computed

        Raises:
            ValueError: If verify_values is True and lists have invalid lengths
            MemoryError: If memory allocation fails
        """
        ...

    @staticmethod
    def from_numpy(
        prices: npt.NDArray[np.float64],
        sizes: npt.NDArray[np.float64],
        norders: Optional[npt.NDArray[np.uint64]] = None,
        verify_values: bool = True,
    ) -> Self:
        """Create a new PyOrderbookLevels instance from numpy arrays.

        Args:
            prices: 1D array of prices
            sizes: 1D array of sizes (must match length of prices)
            norders: Optional 1D array of norders (must match length of prices if provided)
            verify_values: Whether to verify the values

        Returns:
            A new PyOrderbookLevels instance

        Raises:
            ValueError: If verify_values is True and arrays have invalid lengths
            MemoryError: If memory allocation fails
        """
        ...

    @staticmethod
    def from_numpy_with_ticks_and_lots(
        prices: npt.NDArray[np.float64],
        sizes: npt.NDArray[np.float64],
        norders: npt.NDArray[np.uint64],
        tick_size: float,
        lot_size: float,
        verify_values: bool = True,
    ) -> Self:
        """Create a new PyOrderbookLevels instance from numpy arrays with pre-computed ticks and lots.

        Args:
            prices: 1D array of prices
            sizes: 1D array of sizes (must match length of prices)
            norders: 1D array of norders (must match length of prices)
            tick_size: The tick size for computing ticks
            lot_size: The lot size for computing lots
            verify_values: Whether to verify the values

        Returns:
            A new PyOrderbookLevels instance with ticks and lots computed

        Raises:
            ValueError: If verify_values is True and arrays have invalid lengths
            MemoryError: If memory allocation fails
        """
        ...

    # Note: The following methods are cdef-only:
    # @staticmethod
    # cdef PyOrderbookLevels _create(u64 num_levels, OrderbookLevel* levels)
    # @staticmethod
    # cdef PyOrderbookLevels from_ptr(OrderbookLevel* levels_ptr, u64 num_levels)
    # cdef OrderbookLevels to_c_struct(self)

# Factory functions (cdef-only, documented here for reference)
# cdef OrderbookLevel create_orderbook_level(double price, double size, u64 norders=1) noexcept nogil
# cdef OrderbookLevel create_orderbook_level_with_ticks_and_lots(
#     double price, double size, double tick_size, double lot_size, u64 norders=1
# ) noexcept nogil
# cdef OrderbookLevels create_orderbook_levels(u64 num_levels, OrderbookLevel* levels) noexcept nogil

# Memory management (cdef-only)
# cdef void free_orderbook_levels(OrderbookLevels* levels) noexcept nogil

"""Type stubs for enums.pyx - Orderbook sortedness enums."""

from __future__ import annotations

from enum import IntEnum

class PyOrderbookSortedness(IntEnum):
    """Python-facing enum for specifying orderbook level sortedness.

    Use this enum when constructing PyAdvancedOrderbook to indicate
    how incoming snapshot and delta data is sorted.
    """

    UNKNOWN = 0
    """Sortedness is unknown; levels will be sorted on ingestion."""

    ASCENDING = 1
    """Levels are sorted in ascending order (lowest price first)."""

    DESCENDING = 2
    """Levels are sorted in descending order (highest price first)."""

    BIDS_ASCENDING_ASKS_DESCENDING = 3
    """Bids are ascending, asks are descending."""

    BIDS_DESCENDING_ASKS_ASCENDING = 4
    """Bids are descending, asks are ascending (most common exchange format)."""

class CyOrderbookSortedness(IntEnum):
    """Cython-facing enum for specifying orderbook level sortedness.

    This is a cpdef enum that can be used from both Python and Cython.
    For Python code, prefer using PyOrderbookSortedness.
    """

    UNKNOWN = 0
    ASCENDING = 1
    DESCENDING = 2
    BIDS_ASCENDING_ASKS_DESCENDING = 3
    BIDS_DESCENDING_ASKS_ASCENDING = 4

def py_to_cy_orderbook_sortedness(
    sortedness: PyOrderbookSortedness,
) -> CyOrderbookSortedness:
    """Convert a Python-facing sortedness enum to a Cython-facing sortedness enum.

    Args:
        sortedness: Python sortedness enum value

    Returns:
        Corresponding Cython sortedness enum value
    """
    ...

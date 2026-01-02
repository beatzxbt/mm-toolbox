"""
Orderbook sortedness enums for level array ordering.

Defines PyOrderbookSortedness (Python IntEnum) and CyOrderbookSortedness (Cython
enum) with conversion utilities. Used to specify expected sort order of incoming
snapshot/delta levels to optimize or skip sorting operations.
"""
from __future__ import annotations

from enum import IntEnum


class PyOrderbookSortedness(IntEnum):
    UNKNOWN = 0
    ASCENDING = 1
    DESCENDING = 2
    BIDS_ASCENDING_ASKS_DESCENDING = 3
    BIDS_DESCENDING_ASKS_ASCENDING = 4

cpdef CyOrderbookSortedness py_to_cy_orderbook_sortedness(object sortedness):
    """Convert a Python-facing sortedness enum to a Cython-facing sortedness enum."""
    if sortedness == PyOrderbookSortedness.UNKNOWN:
        return CyOrderbookSortedness.UNKNOWN
    elif sortedness == PyOrderbookSortedness.ASCENDING:
        return CyOrderbookSortedness.ASCENDING
    elif sortedness == PyOrderbookSortedness.DESCENDING:
        return CyOrderbookSortedness.DESCENDING
    elif sortedness == PyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING:
        return CyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING
    elif sortedness == PyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING:
        return CyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING
    else:
        return CyOrderbookSortedness.UNKNOWN

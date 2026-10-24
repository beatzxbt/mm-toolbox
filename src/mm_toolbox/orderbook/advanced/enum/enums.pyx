"""
Orderbook sortedness enums for level array ordering.

Defines PyOrderbookSortedness (Python IntEnum) and CyOrderbookSortedness (Cython
enum) with conversion utilities. Used to specify expected sort order of incoming
snapshot/delta levels or let the core infer stable ordering on first use.
"""
from __future__ import annotations

from enum import IntEnum


class PyOrderbookSortedness(IntEnum):
    """Python-facing enum for orderbook level array ordering.

    Attributes:
        UNKNOWN: Infer stable sort order on first input; reject unsorted input.
        ASCENDING: Levels are in ascending order by price.
        DESCENDING: Levels are in descending order by price.
        BIDS_ASCENDING_ASKS_DESCENDING: Bids ascending, asks descending.
        BIDS_DESCENDING_ASKS_ASCENDING: Bids descending, asks ascending.
    """
    UNKNOWN = 0
    ASCENDING = 1
    DESCENDING = 2
    BIDS_ASCENDING_ASKS_DESCENDING = 3
    BIDS_DESCENDING_ASKS_ASCENDING = 4

cpdef CyOrderbookSortedness py_to_cy_orderbook_sortedness(object sortedness):
    """Convert a Python-facing sortedness enum to a Cython-facing sortedness enum."""
    if sortedness == PyOrderbookSortedness.UNKNOWN:
        return CyOrderbookSortedness.UNKNOWN
    if sortedness == PyOrderbookSortedness.ASCENDING:
        return CyOrderbookSortedness.ASCENDING
    if sortedness == PyOrderbookSortedness.DESCENDING:
        return CyOrderbookSortedness.DESCENDING
    if sortedness == PyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING:
        return CyOrderbookSortedness.BIDS_ASCENDING_ASKS_DESCENDING
    if sortedness == PyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING:
        return CyOrderbookSortedness.BIDS_DESCENDING_ASKS_ASCENDING
    return CyOrderbookSortedness.UNKNOWN

"""Fixtures specific to Python wrapper tests.

Provides pre-populated and empty orderbook fixtures for wrapper Python tests.
Re-exports shared fixtures from the parent conftest.
"""

from __future__ import annotations

# Re-export shared fixtures from parent conftest
from tests.orderbook.advanced.conftest import *  # noqa: F403, F401

import pytest
from tests.orderbook.advanced.conftest import (
    _mk_book,
    _make_levels,
)


@pytest.fixture
def standard_book():
    """Pre-populated orderbook with 3 levels per side.

    Returns:
        PyAdvancedOrderbook with bids at 100.0/99.99/99.98 and asks at 100.01/100.02/100.03.
    """
    book = _mk_book(num_levels=64)
    bids, _ = _make_levels(
        prices=[100.0, 99.99, 99.98],
        sizes=[1.0, 2.0, 3.0],
        norders=[1, 2, 3],
        with_precision=True,
    )
    asks, _ = _make_levels(
        prices=[100.01, 100.02, 100.03],
        sizes=[1.5, 2.5, 3.5],
        norders=[1, 2, 3],
        with_precision=True,
    )
    book.consume_snapshot(asks, bids)
    return book


@pytest.fixture
def empty_book():
    """Fresh unpopulated orderbook.

    Returns:
        PyAdvancedOrderbook with no levels.
    """
    return _mk_book(num_levels=64)

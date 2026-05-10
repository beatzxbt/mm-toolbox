"""Wrapper to expose Layer 1 native Cython tests to pytest.

Tests primitive orderbook components: OrderbookLevel creation, tick/lot
conversions, level sorting/reversing, and OrderbookLevels management.
These are Layer 1 (primitive) tests executed via the compiled Cython test module.
"""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
# engine/test_level.py -> engine -> advanced -> orderbook -> tests
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_level as _native
except ImportError as e:
    import pytest

    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 1: Primitives (27 test functions)


def test_create_orderbook_level_basic():
    """Given basic parameters, When creating OrderbookLevel, Then succeeds."""
    _native.test_create_orderbook_level_basic()


def test_create_orderbook_level_default_norders():
    """Given no norders, When creating OrderbookLevel, Then defaults to 1."""
    _native.test_create_orderbook_level_default_norders()


def test_create_orderbook_level_zero_price():
    """Given price=0, When creating OrderbookLevel, Then succeeds."""
    _native.test_create_orderbook_level_zero_price()


def test_create_orderbook_level_zero_size():
    """Given size=0, When creating OrderbookLevel, Then succeeds."""
    _native.test_create_orderbook_level_zero_size()


def test_create_orderbook_level_large_values():
    """Given large values, When creating OrderbookLevel, Then handles correctly."""
    _native.test_create_orderbook_level_large_values()


def test_create_orderbook_level_with_ticks_and_lots_basic():
    """Given tick/lot info, When creating OrderbookLevel, Then computes correctly."""
    _native.test_create_orderbook_level_with_ticks_and_lots_basic()


def test_create_orderbook_level_with_ticks_and_lots_default_norders():
    """Given no norders with ticks/lots, When creating OrderbookLevel, Then defaults to 1."""
    _native.test_create_orderbook_level_with_ticks_and_lots_default_norders()


def test_create_orderbook_level_ticks_rounding():
    """Given prices near tick boundaries, When converting, Then rounds correctly."""
    _native.test_create_orderbook_level_ticks_rounding()


def test_create_orderbook_level_very_small_tick():
    """Given very small tick size, When creating level, Then handles precision."""
    _native.test_create_orderbook_level_very_small_tick()


def test_create_orderbook_level_large_tick():
    """Given large tick size, When creating level, Then handles correctly."""
    _native.test_create_orderbook_level_large_tick()


def test_convert_price_to_tick_basic():
    """Given price, When converting to ticks, Then correct integer returned."""
    _native.test_convert_price_to_tick_basic()


def test_convert_price_from_tick_basic():
    """Given ticks, When converting to price, Then correct float returned."""
    _native.test_convert_price_from_tick_basic()


def test_tick_conversion_roundtrip():
    """Given price, When roundtripped through ticks, Then original price recovered."""
    _native.test_tick_conversion_roundtrip()


def test_convert_size_to_lot_basic():
    """Given size, When converting to lots, Then correct integer returned."""
    _native.test_convert_size_to_lot_basic()


def test_convert_size_from_lot_basic():
    """Given lots, When converting to size, Then correct float returned."""
    _native.test_convert_size_from_lot_basic()


def test_lot_conversion_roundtrip():
    """Given size, When roundtripped through lots, Then original size recovered."""
    _native.test_lot_conversion_roundtrip()


def test_convert_zero_price():
    """Given price=0, When converting, Then ticks=0."""
    _native.test_convert_zero_price()


def test_convert_zero_size():
    """Given size=0, When converting, Then lots=0."""
    _native.test_convert_zero_size()


def test_swap_levels():
    """Given two levels, When swapped, Then positions exchanged."""
    _native.test_swap_levels()


def test_reverse_levels_basic():
    """Given multiple levels, When reversed, Then order inverted."""
    _native.test_reverse_levels_basic()


def test_reverse_levels_single():
    """Given single level, When reversed, Then unchanged."""
    _native.test_reverse_levels_single()


def test_reverse_levels_even():
    """Given even number of levels, When reversed, Then all positions inverted."""
    _native.test_reverse_levels_even()


def test_sort_levels_ascending():
    """Given unsorted levels, When sorted ascending, Then ordered correctly."""
    _native.test_sort_levels_ascending()


def test_sort_levels_descending():
    """Given unsorted levels, When sorted descending, Then ordered correctly."""
    _native.test_sort_levels_descending()


def test_sort_levels_already_sorted():
    """Given sorted levels, When sorted again, Then unchanged."""
    _native.test_sort_levels_already_sorted()


def test_create_orderbook_levels():
    """Given parameters, When creating OrderbookLevels, Then succeeds."""
    _native.test_create_orderbook_levels()


def test_free_orderbook_levels():
    """Given OrderbookLevels, When freed, Then memory released."""
    _native.test_free_orderbook_levels()

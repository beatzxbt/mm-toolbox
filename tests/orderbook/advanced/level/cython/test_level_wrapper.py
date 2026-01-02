"""Wrapper to expose Layer 1 native Cython tests to pytest."""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
# level/cython/test_level_wrapper.py -> level/cython -> level -> advanced -> orderbook -> tests
test_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_level as _native
except ImportError as e:
    import pytest

    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 1: Primitives (27 test functions)


def test_create_orderbook_level_basic():
    _native.test_create_orderbook_level_basic()


def test_create_orderbook_level_default_norders():
    _native.test_create_orderbook_level_default_norders()


def test_create_orderbook_level_zero_price():
    _native.test_create_orderbook_level_zero_price()


def test_create_orderbook_level_zero_size():
    _native.test_create_orderbook_level_zero_size()


def test_create_orderbook_level_large_values():
    _native.test_create_orderbook_level_large_values()


def test_create_orderbook_level_with_ticks_and_lots_basic():
    _native.test_create_orderbook_level_with_ticks_and_lots_basic()


def test_create_orderbook_level_with_ticks_and_lots_default_norders():
    _native.test_create_orderbook_level_with_ticks_and_lots_default_norders()


def test_create_orderbook_level_ticks_rounding():
    _native.test_create_orderbook_level_ticks_rounding()


def test_create_orderbook_level_very_small_tick():
    _native.test_create_orderbook_level_very_small_tick()


def test_create_orderbook_level_large_tick():
    _native.test_create_orderbook_level_large_tick()


def test_convert_price_to_tick_basic():
    _native.test_convert_price_to_tick_basic()


def test_convert_price_from_tick_basic():
    _native.test_convert_price_from_tick_basic()


def test_tick_conversion_roundtrip():
    _native.test_tick_conversion_roundtrip()


def test_convert_size_to_lot_basic():
    _native.test_convert_size_to_lot_basic()


def test_convert_size_from_lot_basic():
    _native.test_convert_size_from_lot_basic()


def test_lot_conversion_roundtrip():
    _native.test_lot_conversion_roundtrip()


def test_convert_zero_price():
    _native.test_convert_zero_price()


def test_convert_zero_size():
    _native.test_convert_zero_size()


def test_swap_levels():
    _native.test_swap_levels()


def test_reverse_levels_basic():
    _native.test_reverse_levels_basic()


def test_reverse_levels_single():
    _native.test_reverse_levels_single()


def test_reverse_levels_even():
    _native.test_reverse_levels_even()


def test_sort_levels_ascending():
    _native.test_sort_levels_ascending()


def test_sort_levels_descending():
    _native.test_sort_levels_descending()


def test_sort_levels_already_sorted():
    _native.test_sort_levels_already_sorted()


def test_create_orderbook_levels():
    _native.test_create_orderbook_levels()


def test_free_orderbook_levels():
    _native.test_free_orderbook_levels()

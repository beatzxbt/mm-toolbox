"""Wrapper to expose Layer 1 native Cython tests to pytest.

Tests primitive orderbook components: raw OrderbookLevel creation, internal
entry conversion, and OrderbookLevels management.
These are Layer 1 (primitive) tests executed via the compiled Cython test module.
"""

from __future__ import annotations

import os
import sys

import pytest

# Go up to tests/ directory to find the compiled .so files
# engine/test_level.py -> engine -> advanced -> orderbook -> tests
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_level as _native
except ImportError as e:
    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 1: Primitives (18 test functions)


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


def test_create_orderbook_levels():
    """Given parameters, When creating OrderbookLevels, Then succeeds."""
    _native.test_create_orderbook_levels()


def test_free_orderbook_levels():
    """Given OrderbookLevels, When freed, Then memory released."""
    _native.test_free_orderbook_levels()


# Layer 1: Validation helpers (7 test functions)


def test_validate_price_valid():
    """Given valid prices, When validated, Then no error raised."""
    _native.test_validate_price_valid()


def test_validate_price_negative_raises():
    """Given negative price, When validated, Then ValueError raised."""
    _native.test_validate_price_negative_raises()


def test_validate_price_nan_raises():
    """Given NaN price, When validated, Then ValueError raised."""
    _native.test_validate_price_nan_raises()


def test_validate_price_inf_raises():
    """Given Inf price, When validated, Then ValueError raised."""
    _native.test_validate_price_inf_raises()


def test_validate_size_valid():
    """Given valid size, When validated, Then no error raised."""
    _native.test_validate_size_valid()


def test_validate_size_negative_raises():
    """Given negative size, When validated, Then ValueError raised."""
    _native.test_validate_size_negative_raises()


def test_validate_size_nan_raises():
    """Given NaN size, When validated, Then ValueError raised."""
    _native.test_validate_size_nan_raises()


def test_validate_size_inf_raises():
    """Given Inf size, When validated, Then ValueError raised."""
    _native.test_validate_size_inf_raises()

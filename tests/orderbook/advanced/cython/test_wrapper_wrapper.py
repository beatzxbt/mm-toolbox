"""Wrapper to expose Layer 4 native Cython tests to pytest."""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
# cython/test_wrapper_wrapper.py -> cython -> advanced -> orderbook -> tests
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_wrapper as _native
except ImportError as e:
    import pytest

    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 4: AdvancedOrderbook Wrapper (12 test functions)


def test_wrapper_init():
    _native.test_wrapper_init()


def test_wrapper_invalid_tick_raises():
    _native.test_wrapper_invalid_tick_raises()


def test_wrapper_consume_snapshot():
    _native.test_wrapper_consume_snapshot()


def test_wrapper_consume_deltas():
    _native.test_wrapper_consume_deltas()


def test_wrapper_consume_bbo():
    _native.test_wrapper_consume_bbo()


def test_wrapper_clear():
    _native.test_wrapper_clear()


def test_wrapper_price_calculations():
    _native.test_wrapper_price_calculations()


def test_wrapper_crossing_and_change():
    _native.test_wrapper_crossing_and_change()

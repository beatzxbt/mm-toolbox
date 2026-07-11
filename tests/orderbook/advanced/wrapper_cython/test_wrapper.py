"""Wrapper to expose Cython wrapper tests to pytest.

Layer 3: Delegates to native cython_test_wrapper module to verify
Cython-level wrapper functionality including init, snapshot/delta/bbo
delegation, calculations, and clear operations.
"""

from __future__ import annotations

import os
import sys

import pytest

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_wrapper as _native
except ImportError as e:
    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


def test_wrapper_init():
    """Given fresh wrapper, When initialized, Then succeeds."""
    _native.test_wrapper_init()


def test_wrapper_consume_snapshot_delegation():
    """Given snapshot data, When consume_snapshot is delegated, Then core receives it."""
    _native.test_wrapper_consume_snapshot_delegation()


def test_wrapper_consume_deltas_delegation():
    """Given delta data, When consume_deltas is delegated, Then core receives it."""
    _native.test_wrapper_consume_deltas_delegation()


def test_wrapper_consume_bbo_delegation():
    """Given BBO data, When consume_bbo is delegated, Then core receives it."""
    _native.test_wrapper_consume_bbo_delegation()


def test_wrapper_consume_bbo_rejects_crossed_input():
    """Given crossed BBO data, When consumed, Then ValueError is raised."""
    _native.test_wrapper_consume_bbo_rejects_crossed_input()


def test_wrapper_calculation_delegation():
    """Given populated book, When calculations are delegated, Then correct values returned."""
    _native.test_wrapper_calculation_delegation()


def test_wrapper_clear_delegation():
    """Given populated book, When clear is delegated, Then book is emptied."""
    _native.test_wrapper_clear_delegation()

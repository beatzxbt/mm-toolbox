"""Wrapper to expose Cython wrapper tests to pytest."""

from __future__ import annotations

import sys
import os

test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_wrapper as _native
except ImportError as e:
    import pytest

    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


def test_wrapper_init():
    _native.test_wrapper_init()


def test_wrapper_consume_snapshot_delegation():
    _native.test_wrapper_consume_snapshot_delegation()


def test_wrapper_consume_deltas_delegation():
    _native.test_wrapper_consume_deltas_delegation()


def test_wrapper_consume_bbo_delegation():
    _native.test_wrapper_consume_bbo_delegation()


def test_wrapper_calculation_delegation():
    _native.test_wrapper_calculation_delegation()


def test_wrapper_clear_delegation():
    _native.test_wrapper_clear_delegation()

"""Wrapper to expose Layer 2 native Cython tests to pytest."""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
# ladder/cython/test_ladder_wrapper.py -> ladder/cython -> ladder -> advanced -> orderbook -> tests
test_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_ladder as _native
except ImportError as e:
    import pytest

    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 2: OrderbookLadder (24 test functions)


# Initialization tests
def test_ladder_init_basic():
    _native.test_ladder_init_basic()


def test_ladder_init_ascending():
    _native.test_ladder_init_ascending()


def test_ladder_init_descending():
    _native.test_ladder_init_descending()


def test_ladder_init_single_level():
    _native.test_ladder_init_single_level()


def test_ladder_init_large():
    _native.test_ladder_init_large()


# Insert tests
def test_ladder_insert_level():
    _native.test_ladder_insert_level()


def test_ladder_insert_multiple():
    _native.test_ladder_insert_multiple()


# Roll right tests
def test_ladder_roll_right_at_start():
    _native.test_ladder_roll_right_at_start()


def test_ladder_roll_right_in_middle():
    _native.test_ladder_roll_right_in_middle()


def test_ladder_roll_right_at_max_capacity():
    _native.test_ladder_roll_right_at_max_capacity()


# Roll left tests
def test_ladder_roll_left_at_start():
    _native.test_ladder_roll_left_at_start()


def test_ladder_roll_left_in_middle():
    _native.test_ladder_roll_left_in_middle()


def test_ladder_roll_left_at_end():
    _native.test_ladder_roll_left_at_end()


# Reset tests
def test_ladder_reset():
    _native.test_ladder_reset()


# State check tests
def test_ladder_is_empty():
    _native.test_ladder_is_empty()


def test_ladder_not_empty_after_insert():
    _native.test_ladder_not_empty_after_insert()


def test_ladder_is_full():
    _native.test_ladder_is_full()


# Count management tests
def test_ladder_increment_count_respects_max():
    _native.test_ladder_increment_count_respects_max()


def test_ladder_decrement_count_respects_zero():
    _native.test_ladder_decrement_count_respects_zero()


# get_data tests
def test_ladder_get_data():
    _native.test_ladder_get_data()


def test_ladder_data_reflects_changes():
    _native.test_ladder_data_reflects_changes()


# NumPy accessor tests
def test_ladder_get_levels():
    _native.test_ladder_get_levels()


def test_ladder_get_prices():
    _native.test_ladder_get_prices()


def test_ladder_get_sizes():
    _native.test_ladder_get_sizes()


def test_ladder_empty_accessors():
    _native.test_ladder_empty_accessors()

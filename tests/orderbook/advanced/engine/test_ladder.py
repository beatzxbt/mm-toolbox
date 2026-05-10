"""Wrapper to expose Layer 2 native Cython tests to pytest.

Tests OrderbookLadder functionality: initialization, insertion, roll
operations, reset, state checks, count management, and NumPy accessors.
These are Layer 2 (composite) tests executed via the compiled Cython test module.
"""

from __future__ import annotations

import os
import sys

import pytest

# Go up to tests/ directory to find the compiled .so files
# engine/test_ladder.py -> engine -> advanced -> orderbook -> tests
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_ladder as _native
except ImportError as e:
    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 2: OrderbookLadder (24 test functions)


# Initialization tests
def test_ladder_init_basic():
    """Given basic params, When initializing ladder, Then succeeds."""
    _native.test_ladder_init_basic()


def test_ladder_init_ascending():
    """Given ascending flag, When initializing ladder, Then set correctly."""
    _native.test_ladder_init_ascending()


def test_ladder_init_descending():
    """Given descending flag, When initializing ladder, Then set correctly."""
    _native.test_ladder_init_descending()


def test_ladder_init_single_level():
    """Given single level capacity, When initializing ladder, Then succeeds."""
    _native.test_ladder_init_single_level()


def test_ladder_init_large():
    """Given large capacity, When initializing ladder, Then allocates correctly."""
    _native.test_ladder_init_large()


# Insert tests
def test_ladder_insert_level():
    """Given a level, When inserted, Then stored correctly."""
    _native.test_ladder_insert_level()


def test_ladder_insert_multiple():
    """Given multiple levels, When inserted, Then all stored correctly."""
    _native.test_ladder_insert_multiple()


# Roll right tests
def test_ladder_roll_right_at_start():
    """Given insertion at start, When rolling right, Then space created."""
    _native.test_ladder_roll_right_at_start()


def test_ladder_roll_right_in_middle():
    """Given insertion in middle, When rolling right, Then elements shifted."""
    _native.test_ladder_roll_right_in_middle()


def test_ladder_roll_right_at_max_capacity():
    """Given full ladder, When rolling right at max capacity, Then handles correctly."""
    _native.test_ladder_roll_right_at_max_capacity()


# Roll left tests
def test_ladder_roll_left_at_start():
    """Given deletion at start, When rolling left, Then elements shifted."""
    _native.test_ladder_roll_left_at_start()


def test_ladder_roll_left_in_middle():
    """Given deletion in middle, When rolling left, Then elements shifted."""
    _native.test_ladder_roll_left_in_middle()


def test_ladder_roll_left_at_end():
    """Given deletion at end, When rolling left, Then handles correctly."""
    _native.test_ladder_roll_left_at_end()


# Reset tests
def test_ladder_reset():
    """Given populated ladder, When reset, Then emptied."""
    _native.test_ladder_reset()


# State check tests
def test_ladder_is_empty():
    """Given empty ladder, When checking is_empty, Then returns True."""
    _native.test_ladder_is_empty()


def test_ladder_not_empty_after_insert():
    """Given insertion, When checking is_empty, Then returns False."""
    _native.test_ladder_not_empty_after_insert()


def test_ladder_is_full():
    """Given full ladder, When checking is_full, Then returns True."""
    _native.test_ladder_is_full()


# Count management tests
def test_ladder_increment_count_respects_max():
    """Given full ladder, When incrementing count, Then respects max."""
    _native.test_ladder_increment_count_respects_max()


def test_ladder_decrement_count_respects_zero():
    """Given empty ladder, When decrementing count, Then respects zero."""
    _native.test_ladder_decrement_count_respects_zero()


# get_data tests
def test_ladder_get_data():
    """Given populated ladder, When getting data, Then returns correct array."""
    _native.test_ladder_get_data()


def test_ladder_data_reflects_changes():
    """Given modified ladder, When getting data, Then reflects changes."""
    _native.test_ladder_data_reflects_changes()


# NumPy accessor tests
def test_ladder_get_levels():
    """Given populated ladder, When getting levels, Then returns correct data."""
    _native.test_ladder_get_levels()


def test_ladder_get_prices():
    """Given populated ladder, When getting prices, Then returns correct array."""
    _native.test_ladder_get_prices()


def test_ladder_get_sizes():
    """Given populated ladder, When getting sizes, Then returns correct array."""
    _native.test_ladder_get_sizes()


def test_ladder_empty_accessors():
    """Given empty ladder, When accessing data, Then handles correctly."""
    _native.test_ladder_empty_accessors()

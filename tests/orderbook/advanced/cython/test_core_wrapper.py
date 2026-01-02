"""Wrapper to expose Layer 3 native Cython tests to pytest."""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
# cython/test_core_wrapper.py -> cython -> advanced -> orderbook -> tests
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_core as _native
except ImportError as e:
    import pytest

    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# Layer 3: CoreAdvancedOrderbook (85 test functions)


def test_core_init_valid():
    _native.test_core_init_valid()


def test_core_init_zero_tick_size():
    _native.test_core_init_zero_tick_size()


def test_core_init_negative_tick_size():
    _native.test_core_init_negative_tick_size()


def test_core_init_zero_lot_size():
    _native.test_core_init_zero_lot_size()


def test_core_init_negative_lot_size():
    _native.test_core_init_negative_lot_size()


def test_core_init_zero_levels():
    _native.test_core_init_zero_levels()


def test_core_init_single_level():
    _native.test_core_init_single_level()


def test_core_init_large_levels():
    _native.test_core_init_large_levels()


def test_core_init_all_sortedness_modes():
    _native.test_core_init_all_sortedness_modes()


def test_core_snapshot_basic():
    _native.test_core_snapshot_basic()


def test_core_snapshot_replaces_existing():
    _native.test_core_snapshot_replaces_existing()


def test_core_snapshot_more_levels_than_max():
    _native.test_core_snapshot_more_levels_than_max()


def test_core_snapshot_fewer_levels_than_max():
    _native.test_core_snapshot_fewer_levels_than_max()


def test_core_snapshot_single_level_each():
    _native.test_core_snapshot_single_level_each()


def test_core_snapshot_sortedness_unknown():
    _native.test_core_snapshot_sortedness_unknown()


def test_core_snapshot_populates_ticks_and_lots():
    _native.test_core_snapshot_populates_ticks_and_lots()


def test_core_snapshot_overwrites_ticks_and_lots():
    _native.test_core_snapshot_overwrites_ticks_and_lots()


def test_core_delta_ask_consume_bbo_size():
    _native.test_core_delta_ask_consume_bbo_size()


def test_core_delta_ask_delete_bbo():
    _native.test_core_delta_ask_delete_bbo()


def test_core_delta_ask_insert_new_bbo():
    _native.test_core_delta_ask_insert_new_bbo()


def test_core_delta_ask_insert_new_bbo_removes_overlapping_bids():
    _native.test_core_delta_ask_insert_new_bbo_removes_overlapping_bids()


def test_core_delta_ask_insert_middle():
    _native.test_core_delta_ask_insert_middle()


def test_core_delta_ask_update_middle():
    _native.test_core_delta_ask_update_middle()


def test_core_delta_ask_delete_middle():
    _native.test_core_delta_ask_delete_middle()


def test_core_delta_ask_beyond_worst_full_book():
    _native.test_core_delta_ask_beyond_worst_full_book()


def test_core_delta_ask_delete_nonexistent():
    _native.test_core_delta_ask_delete_nonexistent()


def test_core_delta_ask_multiple_sequential():
    _native.test_core_delta_ask_multiple_sequential()


def test_core_delta_bid_consume_bbo_size():
    _native.test_core_delta_bid_consume_bbo_size()


def test_core_delta_bid_delete_bbo():
    _native.test_core_delta_bid_delete_bbo()


def test_core_delta_bid_insert_new_bbo():
    _native.test_core_delta_bid_insert_new_bbo()


def test_core_delta_bid_insert_new_bbo_removes_overlapping_asks():
    _native.test_core_delta_bid_insert_new_bbo_removes_overlapping_asks()


def test_core_delta_bid_update_middle():
    _native.test_core_delta_bid_update_middle()


def test_core_delta_bid_delete_middle():
    _native.test_core_delta_bid_delete_middle()


def test_core_delta_both_sides():
    _native.test_core_delta_both_sides()


def test_core_delta_empty_arrays():
    _native.test_core_delta_empty_arrays()


def test_core_delta_on_empty_book():
    _native.test_core_delta_on_empty_book()


def test_core_delta_deplete_entire_side():
    _native.test_core_delta_deplete_entire_side()


def test_core_bbo_update_same_tick():
    _native.test_core_bbo_update_same_tick()


def test_core_bbo_delete_matching():
    _native.test_core_bbo_delete_matching()


def test_core_bbo_insert_tighter_ask():
    _native.test_core_bbo_insert_tighter_ask()


def test_core_bbo_insert_tighter_bid():
    _native.test_core_bbo_insert_tighter_bid()


def test_core_bbo_crossed_book_resolution():
    _native.test_core_bbo_crossed_book_resolution()


def test_core_bbo_on_empty_book():
    _native.test_core_bbo_on_empty_book()


def test_core_bbo_populates_ticks_and_lots():
    _native.test_core_bbo_populates_ticks_and_lots()


def test_core_mid_price_standard():
    _native.test_core_mid_price_standard()


def test_core_mid_price_1_tick_spread():
    _native.test_core_mid_price_1_tick_spread()


def test_core_mid_price_wide_spread():
    _native.test_core_mid_price_wide_spread()


def test_core_mid_price_empty_raises():
    _native.test_core_mid_price_empty_raises()


def test_core_spread_1_tick():
    _native.test_core_spread_1_tick()


def test_core_spread_multi_tick():
    _native.test_core_spread_multi_tick()


def test_core_spread_empty_raises():
    _native.test_core_spread_empty_raises()


def test_core_wmid_equal_volumes():
    _native.test_core_wmid_equal_volumes()


def test_core_wmid_bid_heavy():
    _native.test_core_wmid_bid_heavy()


def test_core_wmid_empty_raises():
    _native.test_core_wmid_empty_raises()


def test_core_vwmp_zero_size():
    _native.test_core_vwmp_zero_size()


def test_core_vwmp_negative_size():
    _native.test_core_vwmp_negative_size()


def test_core_vwmp_small_size():
    _native.test_core_vwmp_small_size()


def test_core_vwmp_exceeds_liquidity():
    _native.test_core_vwmp_exceeds_liquidity()


def test_core_impact_zero_size():
    _native.test_core_impact_zero_size()


def test_core_impact_negative_size():
    _native.test_core_impact_negative_size()


def test_core_impact_buy_single_level():
    _native.test_core_impact_buy_single_level()


def test_core_impact_sell_single_level():
    _native.test_core_impact_sell_single_level()


def test_core_impact_buy_multi_level():
    _native.test_core_impact_buy_multi_level()


def test_core_impact_exceeds_liquidity():
    _native.test_core_impact_exceeds_liquidity()


def test_core_impact_empty_raises():
    _native.test_core_impact_empty_raises()


def test_core_is_crossed_no_cross():
    _native.test_core_is_crossed_no_cross()


def test_core_is_crossed_bid_crosses_ask():
    _native.test_core_is_crossed_bid_crosses_ask()


def test_core_is_crossed_ask_crosses_bid():
    _native.test_core_is_crossed_ask_crosses_bid()


def test_core_is_crossed_empty_raises():
    _native.test_core_is_crossed_empty_raises()


def test_core_bbo_change_no_change():
    _native.test_core_bbo_change_no_change()


def test_core_bbo_change_bid_differs():
    _native.test_core_bbo_change_bid_differs()


def test_core_bbo_change_ask_differs():
    _native.test_core_bbo_change_ask_differs()


def test_core_bbo_change_both_differ():
    _native.test_core_bbo_change_both_differ()


def test_core_bbo_change_empty_raises():
    _native.test_core_bbo_change_empty_raises()


def test_core_clear():
    _native.test_core_clear()


def test_core_operations_after_clear_raise():
    _native.test_core_operations_after_clear_raise()


def test_core_repopulate_after_clear():
    _native.test_core_repopulate_after_clear()


def test_core_view_accessors():
    _native.test_core_view_accessors()


def test_core_views_reflect_mutations():
    _native.test_core_views_reflect_mutations()


def test_core_rapid_insert_delete():
    _native.test_core_rapid_insert_delete()


def test_core_fill_to_max_then_insert():
    _native.test_core_fill_to_max_then_insert()


def test_core_empty_full_empty_cycle():
    _native.test_core_empty_full_empty_cycle()


def test_core_very_small_tick_size():
    _native.test_core_very_small_tick_size()


def test_core_very_large_tick_size():
    _native.test_core_very_large_tick_size()


def test_core_asymmetric_depths():
    _native.test_core_asymmetric_depths()

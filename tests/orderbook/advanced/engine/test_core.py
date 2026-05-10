"""Wrapper to expose Layer 3 native Cython tests to pytest.

Tests CoreAdvancedOrderbook functionality: initialization, snapshots, deltas,
BBO updates, calculations (mid, spread, wmid, vwmp, impact), cross detection,
clear/repopulate, view accessors, and stress scenarios.
These are Layer 3 (integration) tests executed via the compiled Cython test module.
"""

from __future__ import annotations

import sys
import os

# Go up to tests/ directory to find the compiled .so files
# engine/test_core.py -> engine -> advanced -> orderbook -> tests
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
    """Given valid params, When initializing core, Then succeeds."""
    _native.test_core_init_valid()


def test_core_init_zero_tick_size():
    """Given tick_size=0, When initializing core, Then raises error."""
    _native.test_core_init_zero_tick_size()


def test_core_init_negative_tick_size():
    """Given negative tick_size, When initializing core, Then raises error."""
    _native.test_core_init_negative_tick_size()


def test_core_init_zero_lot_size():
    """Given lot_size=0, When initializing core, Then raises error."""
    _native.test_core_init_zero_lot_size()


def test_core_init_negative_lot_size():
    """Given negative lot_size, When initializing core, Then raises error."""
    _native.test_core_init_negative_lot_size()


def test_core_init_zero_levels():
    """Given num_levels=0, When initializing core, Then raises error."""
    _native.test_core_init_zero_levels()


def test_core_init_single_level():
    """Given num_levels=1, When initializing core, Then succeeds."""
    _native.test_core_init_single_level()


def test_core_init_large_levels():
    """Given large num_levels, When initializing core, Then succeeds."""
    _native.test_core_init_large_levels()


def test_core_init_all_sortedness_modes():
    """Given all sortedness modes, When initializing core, Then succeeds."""
    _native.test_core_init_all_sortedness_modes()


def test_core_snapshot_basic():
    """Given snapshot data, When consumed, Then book populated correctly."""
    _native.test_core_snapshot_basic()


def test_core_snapshot_replaces_existing():
    """Given existing book, When snapshot consumed, Then replaces existing."""
    _native.test_core_snapshot_replaces_existing()


def test_core_snapshot_more_levels_than_max():
    """Given more levels than capacity, When snapshot consumed, Then truncates."""
    _native.test_core_snapshot_more_levels_than_max()


def test_core_snapshot_fewer_levels_than_max():
    """Given fewer levels than capacity, When snapshot consumed, Then stores all."""
    _native.test_core_snapshot_fewer_levels_than_max()


def test_core_snapshot_single_level_each():
    """Given single level per side, When snapshot consumed, Then stores correctly."""
    _native.test_core_snapshot_single_level_each()


def test_core_snapshot_sortedness_unknown():
    """Given unknown sortedness, When snapshot consumed, Then sorts automatically."""
    _native.test_core_snapshot_sortedness_unknown()


def test_core_snapshot_populates_ticks_and_lots():
    """Given snapshot without ticks/lots, When consumed, Then computes them."""
    _native.test_core_snapshot_populates_ticks_and_lots()


def test_core_snapshot_overwrites_ticks_and_lots():
    """Given snapshot with existing ticks/lots, When consumed, Then overwrites."""
    _native.test_core_snapshot_overwrites_ticks_and_lots()


def test_core_delta_ask_consume_bbo_size():
    """Given ask delta at BBO, When consumed, Then updates best ask size."""
    _native.test_core_delta_ask_consume_bbo_size()


def test_core_delta_ask_delete_bbo():
    """Given ask delta deleting BBO, When consumed, Then removes best ask."""
    _native.test_core_delta_ask_delete_bbo()


def test_core_delta_ask_insert_new_bbo():
    """Given ask delta with new better price, When consumed, Then inserts new BBO."""
    _native.test_core_delta_ask_insert_new_bbo()


def test_core_delta_ask_insert_new_bbo_removes_overlapping_bids():
    """Given ask delta crossing bids, When consumed, Then removes overlapping bids."""
    _native.test_core_delta_ask_insert_new_bbo_removes_overlapping_bids()


def test_core_delta_ask_insert_middle():
    """Given ask delta in middle of book, When consumed, Then inserts correctly."""
    _native.test_core_delta_ask_insert_middle()


def test_core_delta_ask_update_middle():
    """Given ask delta updating middle level, When consumed, Then updates correctly."""
    _native.test_core_delta_ask_update_middle()


def test_core_delta_ask_delete_middle():
    """Given ask delta deleting middle level, When consumed, Then removes correctly."""
    _native.test_core_delta_ask_delete_middle()


def test_core_delta_ask_beyond_worst_full_book():
    """Given ask delta beyond worst level on full book, When consumed, Then ignored."""
    _native.test_core_delta_ask_beyond_worst_full_book()


def test_core_delta_ask_delete_nonexistent():
    """Given ask delta deleting non-existent level, When consumed, Then no-op."""
    _native.test_core_delta_ask_delete_nonexistent()


def test_core_delta_ask_multiple_sequential():
    """Given multiple sequential ask deltas, When consumed, Then all applied correctly."""
    _native.test_core_delta_ask_multiple_sequential()


def test_core_delta_bid_consume_bbo_size():
    """Given bid delta at BBO, When consumed, Then updates best bid size."""
    _native.test_core_delta_bid_consume_bbo_size()


def test_core_delta_bid_delete_bbo():
    """Given bid delta deleting BBO, When consumed, Then removes best bid."""
    _native.test_core_delta_bid_delete_bbo()


def test_core_delta_bid_insert_new_bbo():
    """Given bid delta with new better price, When consumed, Then inserts new BBO."""
    _native.test_core_delta_bid_insert_new_bbo()


def test_core_delta_bid_insert_new_bbo_removes_overlapping_asks():
    """Given bid delta crossing asks, When consumed, Then removes overlapping asks."""
    _native.test_core_delta_bid_insert_new_bbo_removes_overlapping_asks()


def test_core_delta_bid_update_middle():
    """Given bid delta updating middle level, When consumed, Then updates correctly."""
    _native.test_core_delta_bid_update_middle()


def test_core_delta_bid_delete_middle():
    """Given bid delta deleting middle level, When consumed, Then removes correctly."""
    _native.test_core_delta_bid_delete_middle()


def test_core_delta_both_sides():
    """Given deltas for both sides, When consumed, Then both updated correctly."""
    _native.test_core_delta_both_sides()


def test_core_delta_empty_arrays():
    """Given empty delta arrays, When consumed, Then no-op."""
    _native.test_core_delta_empty_arrays()


def test_core_delta_on_empty_book():
    """Given empty book, When delta consumed, Then handles correctly."""
    _native.test_core_delta_on_empty_book()


def test_core_delta_deplete_entire_side():
    """Given delta deleting all levels on one side, When consumed, Then side emptied."""
    _native.test_core_delta_deplete_entire_side()


def test_core_bbo_update_same_tick():
    """Given BBO update at same tick, When consumed, Then updates size."""
    _native.test_core_bbo_update_same_tick()


def test_core_bbo_delete_matching():
    """Given BBO deletion matching existing, When consumed, Then removes level."""
    _native.test_core_bbo_delete_matching()


def test_core_bbo_insert_tighter_ask():
    """Given tighter ask BBO, When consumed, Then inserts and prunes stale."""
    _native.test_core_bbo_insert_tighter_ask()


def test_core_bbo_insert_tighter_bid():
    """Given tighter bid BBO, When consumed, Then inserts and prunes stale."""
    _native.test_core_bbo_insert_tighter_bid()


def test_core_bbo_crossed_book_resolution():
    """Given crossed BBO, When consumed, Then resolves by removing crossed levels."""
    _native.test_core_bbo_crossed_book_resolution()


def test_core_bbo_on_empty_book():
    """Given empty book, When BBO consumed, Then populates both sides."""
    _native.test_core_bbo_on_empty_book()


def test_core_bbo_populates_ticks_and_lots():
    """Given BBO without ticks/lots, When consumed, Then computes them."""
    _native.test_core_bbo_populates_ticks_and_lots()


def test_core_mid_price_standard():
    """Given standard book, When get_mid_price called, Then returns correct value."""
    _native.test_core_mid_price_standard()


def test_core_mid_price_1_tick_spread():
    """Given 1-tick spread, When get_mid_price called, Then handles correctly."""
    _native.test_core_mid_price_1_tick_spread()


def test_core_mid_price_wide_spread():
    """Given wide spread, When get_mid_price called, Then returns correct value."""
    _native.test_core_mid_price_wide_spread()


def test_core_mid_price_empty_raises():
    """Given empty book, When get_mid_price called, Then raises error."""
    _native.test_core_mid_price_empty_raises()


def test_core_spread_1_tick():
    """Given 1-tick spread, When get_bbo_spread called, Then returns tick_size."""
    _native.test_core_spread_1_tick()


def test_core_spread_multi_tick():
    """Given multi-tick spread, When get_bbo_spread called, Then correct value."""
    _native.test_core_spread_multi_tick()


def test_core_spread_empty_raises():
    """Given empty book, When get_bbo_spread called, Then raises error."""
    _native.test_core_spread_empty_raises()


def test_core_wmid_equal_volumes():
    """Given equal volumes, When get_wmid_price called, Then equals mid price."""
    _native.test_core_wmid_equal_volumes()


def test_core_wmid_bid_heavy():
    """Given bid-heavy book, When get_wmid_price called, Then skewed toward bid."""
    _native.test_core_wmid_bid_heavy()


def test_core_wmid_empty_raises():
    """Given empty book, When get_wmid_price called, Then raises error."""
    _native.test_core_wmid_empty_raises()


def test_core_vwmp_zero_size():
    """Given size=0, When get_volume_weighted_mid_price called, Then returns mid."""
    _native.test_core_vwmp_zero_size()


def test_core_vwmp_negative_size():
    """Given negative size, When get_volume_weighted_mid_price called, Then handles."""
    _native.test_core_vwmp_negative_size()


def test_core_vwmp_small_size():
    """Given small size, When get_volume_weighted_mid_price called, Then within first level."""
    _native.test_core_vwmp_small_size()


def test_core_vwmp_exceeds_liquidity():
    """Given size exceeding liquidity, When get_volume_weighted_mid_price called, Then handles."""
    _native.test_core_vwmp_exceeds_liquidity()


def test_core_impact_zero_size():
    """Given size=0, When get_price_impact called, Then returns 0."""
    _native.test_core_impact_zero_size()


def test_core_impact_negative_size():
    """Given negative size, When get_price_impact called, Then handles."""
    _native.test_core_impact_negative_size()


def test_core_impact_buy_single_level():
    """Given buy within single level, When get_price_impact called, Then returns 0."""
    _native.test_core_impact_buy_single_level()


def test_core_impact_sell_single_level():
    """Given sell within single level, When get_price_impact called, Then returns 0."""
    _native.test_core_impact_sell_single_level()


def test_core_impact_buy_multi_level():
    """Given buy spanning multiple levels, When get_price_impact called, Then terminal impact."""
    _native.test_core_impact_buy_multi_level()


def test_core_impact_exceeds_liquidity():
    """Given size exceeding liquidity, When get_price_impact called, Then returns inf."""
    _native.test_core_impact_exceeds_liquidity()


def test_core_impact_empty_raises():
    """Given empty book, When get_price_impact called, Then raises error."""
    _native.test_core_impact_empty_raises()


def test_core_is_crossed_no_cross():
    """Given normal book, When is_crossed checked, Then returns False."""
    _native.test_core_is_crossed_no_cross()


def test_core_is_crossed_bid_crosses_ask():
    """Given bid > ask, When is_crossed checked, Then returns True."""
    _native.test_core_is_crossed_bid_crosses_ask()


def test_core_is_crossed_ask_crosses_bid():
    """Given ask < bid, When is_crossed checked, Then returns True."""
    _native.test_core_is_crossed_ask_crosses_bid()


def test_core_is_crossed_empty_raises():
    """Given empty book, When is_crossed checked, Then raises error."""
    _native.test_core_is_crossed_empty_raises()


def test_core_bbo_change_no_change():
    """Given same BBO prices, When checked, Then no change detected."""
    _native.test_core_bbo_change_no_change()


def test_core_bbo_change_bid_differs():
    """Given different bid price, When checked, Then change detected."""
    _native.test_core_bbo_change_bid_differs()


def test_core_bbo_change_ask_differs():
    """Given different ask price, When checked, Then change detected."""
    _native.test_core_bbo_change_ask_differs()


def test_core_bbo_change_both_differ():
    """Given both prices differ, When checked, Then change detected."""
    _native.test_core_bbo_change_both_differ()


def test_core_bbo_change_empty_raises():
    """Given empty book, When BBO change checked, Then raises error."""
    _native.test_core_bbo_change_empty_raises()


def test_core_clear():
    """Given populated book, When cleared, Then emptied."""
    _native.test_core_clear()


def test_core_operations_after_clear_raise():
    """Given cleared book, When operations called, Then raise error."""
    _native.test_core_operations_after_clear_raise()


def test_core_repopulate_after_clear():
    """Given cleared book, When repopulated, Then works normally."""
    _native.test_core_repopulate_after_clear()


def test_core_view_accessors():
    """Given populated book, When view accessors called, Then return correct data."""
    _native.test_core_view_accessors()


def test_core_views_reflect_mutations():
    """Given modified book, When views accessed, Then reflect mutations."""
    _native.test_core_views_reflect_mutations()


def test_core_rapid_insert_delete():
    """Given rapid operations, When executed, Then maintains consistency."""
    _native.test_core_rapid_insert_delete()


def test_core_fill_to_max_then_insert():
    """Given full book, When inserting better level, Then evicts worst."""
    _native.test_core_fill_to_max_then_insert()


def test_core_empty_full_empty_cycle():
    """Given book, When cycled empty-full-empty, Then handles correctly."""
    _native.test_core_empty_full_empty_cycle()


def test_core_very_small_tick_size():
    """Given very small tick size, When operations performed, Then handles precision."""
    _native.test_core_very_small_tick_size()


def test_core_very_large_tick_size():
    """Given very large tick size, When operations performed, Then handles correctly."""
    _native.test_core_very_large_tick_size()


def test_core_asymmetric_depths():
    """Given asymmetric bid/ask depths, When operations performed, Then handles correctly."""
    _native.test_core_asymmetric_depths()

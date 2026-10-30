"""Readable exchange-event traces and deterministic, replayable event generation."""

from dataclasses import dataclass, replace
from decimal import Decimal
from random import Random

BASE_BIDS = [(100.0, 10.0, 1), (99.0, 20.0, 2), (98.0, 30.0, 3), (97.0, 40.0, 4)]
BASE_ASKS = [(105.0, 15.0, 5), (106.0, 25.0, 6), (107.0, 35.0, 7), (108.0, 45.0, 8)]


@dataclass(frozen=True)
class Event:
    kind: str
    asks: tuple = ()
    bids: tuple = ()
    reject: bool = False
    name: str = ""
    args: tuple = ()
    expected: object = None
    error: str | None = None


def snapshot(asks=BASE_ASKS, bids=BASE_BIDS, reject=False):
    return Event("snapshot", tuple(asks), tuple(bids), reject)


def delta(asks=(), bids=(), reject=False):
    return Event("deltas", tuple(asks), tuple(bids), reject)


def bbo(ask, bid, reject=False, error=None):
    return Event("bbo", tuple(ask), tuple(bid), reject, error=error)


def query(name, *args, expected=None):
    return Event("query", name=name, args=args, expected=expected)


@dataclass(frozen=True)
class Scenario:
    id: str
    events: tuple
    capacity: int = 64
    sortedness: int = 0
    input_sortedness: int | None = None
    tick: float = 1.0
    lot: float = 1.0


def _trace(identifier, *events, **kwargs):
    return Scenario(identifier, (snapshot(), *events), **kwargs)


def scenarios():
    cases = [
        Scenario("S1_initialize_full_depth", (snapshot(),)),
        _trace(
            "S2_snapshot_replaces_whole_book",
            snapshot([(205.0, 8.0, 8)], [(200.0, 9.0, 9)]),
        ),
        _trace(
            "S3_snapshot_duplicates_last_absolute_wins",
            snapshot(BASE_ASKS + [(105.0, 7.0, 9)], BASE_BIDS + [(100.0, 8.0, 10)]),
        ),
        _trace(
            "S4_snapshot_zeros_and_final_zero_duplicates",
            snapshot(
                BASE_ASKS + [(105.0, 0.0, 0), (109.0, 0.0, 0)],
                BASE_BIDS + [(100.0, 0.0, 0), (96.0, 0.0, 0)],
            ),
        ),
        _trace(
            "S6_crossed_snapshot_atomic_then_recover",
            snapshot([(99.0, 8.0, 1)], [(100.0, 7.0, 2)], True),
            snapshot([(205.0, 8.0, 3)], [(200.0, 7.0, 4)]),
        ),
        _trace(
            "S6_locked_snapshot_atomic",
            snapshot([(100.0, 8.0, 1)], [(100.0, 7.0, 2)], True),
        ),
        _trace(
            "D1_absolute_quantity_and_order_replacement", delta(bids=[(99.0, 7.0, 9)])
        ),
        _trace(
            "D2_better_worse_and_gap_insertions",
            delta(
                [(103.0, 8.0, 11), (109.0, 9.0, 12)],
                [(102.0, 6.0, 13), (90.0, 4.0, 14)],
            ),
            delta([(104.0, 7.0, 15)], [(101.0, 3.0, 16)]),
        ),
        _trace(
            "D3_delete_best_middle_worst",
            delta([(105.0, 0.0, 0)], [(100.0, 0.0, 0)]),
            delta([(107.0, 0.0, 0)], [(98.0, 0.0, 0)]),
            delta([(108.0, 0.0, 0)], [(97.0, 0.0, 0)]),
        ),
        _trace("D4_absent_zero_noop", delta([(150.0, 0.0, 0)], [(50.0, 0.0, 0)])),
        _trace(
            "D5_empty_and_one_sided_batches",
            delta(),
            delta(asks=[(106.0, 7.0, 9)]),
            delta(bids=[(99.0, 12.0, 10)]),
        ),
        _trace(
            "D6_duplicate_final_nonzero",
            delta(bids=[(99.0, 7.0, 7), (99.0, 0.0, 0), (99.0, 12.0, 12)]),
        ),
        _trace(
            "D6_duplicate_final_zero",
            delta(asks=[(106.0, 7.0, 7), (106.0, 12.0, 12), (106.0, 0.0, 0)]),
        ),
        _trace(
            "D7_delete_restore_across_calls",
            delta(bids=[(99.0, 0.0, 0)]),
            delta(bids=[(99.0, 7.0, 9)]),
        ),
        _trace(
            "D8_temporary_cross_final_zero_preserves_opposite",
            delta(asks=[(99.0, 7.0, 9), (99.0, 0.0, 0)]),
        ),
        _trace(
            "D9_both_sides_move_together", delta([(96.0, 8.0, 11)], [(95.0, 7.0, 12)])
        ),
        _trace(
            "D10_partial_cross_prunes_only_crossed_bids", delta(asks=[(99.0, 8.0, 11)])
        ),
        _trace(
            "D11_full_opposite_wipe_ignored",
            delta(asks=[(96.0, 8.0, 11)]),
            delta(bids=[(109.0, 9.0, 12)]),
        ),
        _trace(
            "D11_absent_zero_cannot_mask_live_ask_wipe",
            delta(asks=[(95.0, 0.0, 0), (96.0, 8.0, 9)]),
        ),
        _trace(
            "D11_absent_zero_cannot_mask_live_bid_wipe",
            delta(bids=[(109.0, 8.0, 9), (110.0, 0.0, 0)]),
        ),
        _trace(
            "C5_cross_cleanup_precedes_capacity",
            delta([(99.0, 8.0, 9)], [(101.0, 7.0, 10)]),
            capacity=4,
        ),
        _trace(
            "B1_same_touch_refreshes_metadata", bbo((105.0, 7.0, 9), (100.0, 8.0, 10))
        ),
        _trace("B2_tighten", bbo((103.0, 7.0, 9), (102.0, 8.0, 10))),
        _trace(
            "B3_outward_one_level_replaces_metadata",
            bbo((106.0, 7.0, 9), (99.0, 8.0, 10)),
        ),
        _trace("B4_outward_multiple_levels", bbo((108.0, 7.0, 9), (97.0, 8.0, 10))),
        Scenario(
            "B5_outward_gaps",
            (
                snapshot(
                    [(105.0, 15.0, 5), (107.0, 25.0, 6), (111.0, 35.0, 7)],
                    [(100.0, 10.0, 1), (98.0, 20.0, 2), (94.0, 30.0, 3)],
                ),
                bbo((106.0, 7.0, 9), (99.0, 8.0, 10)),
            ),
        ),
        _trace("B6_outside_all_existing_depth", bbo((120.0, 7.0, 9), (90.0, 8.0, 10))),
        _trace(
            "B7_shift_down_then_up",
            bbo((98.0, 7.0, 9), (97.0, 8.0, 10)),
            bbo((108.0, 6.0, 11), (107.0, 5.0, 12)),
        ),
        _trace(
            "B8_repeated_bbo_idempotent",
            bbo((106.0, 7.0, 9), (99.0, 8.0, 10)),
            bbo((106.0, 7.0, 9), (99.0, 8.0, 10)),
        ),
        _trace(
            "B9_only_explicit_delta_restores_pruned_levels",
            bbo((107.0, 7.0, 9), (98.0, 8.0, 10)),
            delta([(106.0, 2.0, 11)], [(99.0, 3.0, 12)]),
        ),
        _trace(
            "B10_matching_zero_deletes_touch", bbo((105.0, 0.0, 0), (100.0, 0.0, 0))
        ),
        _trace(
            "B10_unmatched_zero_bid_rejects_atomically",
            bbo((106.0, 7.0, 9), (99.0, 0.0, 0), True),
        ),
        _trace(
            "B10_unmatched_zero_ask_rejects_atomically",
            bbo((106.0, 0.0, 0), (99.0, 7.0, 9), True),
        ),
        Scenario(
            "C1_snapshot_normalizes_before_capacity",
            (
                snapshot(
                    [(105.2, 1.0, 1), (105.1, 9.0, 9)]
                    + [(float(p), float(p), p) for p in range(106, 112)],
                    [(100.2, 1.0, 1), (100.1, 9.0, 9)]
                    + [(float(p), float(p), p) for p in range(99, 93, -1)],
                ),
            ),
            capacity=4,
        ),
        _trace(
            "C2_better_insert_eviction_is_permanent",
            delta([(104.0, 8.0, 9)], [(101.0, 7.0, 10)]),
            delta([(104.0, 0.0, 0)], [(101.0, 0.0, 0)]),
            capacity=4,
        ),
        _trace(
            "C3_worse_insert_bounded_vs_unbounded",
            delta([(120.0, 8.0, 9)], [(90.0, 7.0, 10)]),
            capacity=4,
        ),
        _trace(
            "C4_replace_at_capacity_no_eviction",
            delta([(105.0, 8.0, 9)], [(100.0, 7.0, 10)]),
            capacity=4,
        ),
        _trace(
            "C7_clear_reseed_repeat",
            Event("clear"),
            snapshot(),
            Event("clear"),
            snapshot(),
        ),
        _trace(
            "C8_deplete_ask_restore_delta",
            delta([(p, 0.0, 0) for p, _, _ in BASE_ASKS]),
            delta(asks=[(106.0, 7.0, 9)]),
        ),
        _trace(
            "C8_deplete_bid_restore_bbo",
            delta(bids=[(p, 0.0, 0) for p, _, _ in BASE_BIDS]),
            bbo((106.0, 7.0, 9), (99.0, 8.0, 10)),
        ),
        Scenario(
            "C8_one_side_empty_other_side_changes_then_zero_bbo",
            (
                snapshot([(101.0, 1.0, 1)], [(99.0, 1.0, 1)]),
                delta([(101.0, 0.0, 0)]),
                delta(bids=[(100.0, 1.0, 2)]),
                bbo((102.0, 1.0, 3), (100.0, 0.0, 0)),
            ),
        ),
        _trace(
            "C9_snapshot_recovery_after_depletion",
            delta(bids=[(p, 0.0, 0) for p, _, _ in BASE_BIDS]),
            snapshot(),
        ),
        _trace(
            "M1_spread_and_floored_mid",
            query("get_bbo_spread", expected=5.0),
            query("get_mid_price", expected=102.0),
        ),
        _trace(
            "M2_weighted_mid_and_unequal_touch_quantities",
            query("get_wmid_price", expected=103.0),
            delta([(105.0, 2.0, 9)], [(100.0, 18.0, 10)]),
            query("get_wmid_price", expected=100.0),
        ),
        _trace(
            "M3_volume_weighted_mid",
            query("get_volume_weighted_mid_price", 10.0, True, expected=102.5),
            query("get_volume_weighted_mid_price", 20.0, True, expected=102.375),
        ),
        _trace(
            "M4_exact_exhaustion_and_positive_infinity",
            query("get_volume_weighted_mid_price", 100.0, True),
            query("get_volume_weighted_mid_price", 101.0, True, expected=float("inf")),
            query("get_price_impact", 120.0, True, True),
            query("get_price_impact", 121.0, True, True, expected=float("inf")),
            query("get_price_impact", 100.0, False, True),
            query("get_price_impact", 101.0, False, True, expected=float("inf")),
        ),
        _trace(
            "M5_touch_price_impact_base_and_quote",
            *[
                query(
                    "get_price_impact",
                    amount * (anchor if not base else 1),
                    buy,
                    base,
                    expected=impact,
                )
                for buy, anchor, amount, impact in [
                    (True, 105, 15, 0),
                    (True, 105, 16, 1),
                    (False, 100, 10, 0),
                    (False, 100, 11, 1),
                ]
                for base in (True, False)
            ],
        ),
        Scenario(
            "M6_impact_band_exact_below_above_boundaries",
            (
                snapshot(
                    [(200.0, 15.0, 5), (202.0, 25.0, 6), (204.0, 35.0, 7)],
                    [(100.0, 10.0, 1), (99.0, 20.0, 2), (98.0, 30.0, 3)],
                ),
                *[
                    query("get_size_for_price_impact_bps", 100.0 + offset, buy, base)
                    for buy in (True, False)
                    for offset in (-1e-6, 0.0, 1e-6)
                    for base in (True, False)
                ],
            ),
        ),
        _trace(
            "M7_price_change_and_strict_crossing",
            *[
                query(name, bp, ap, expected=value)
                for name, bp, ap, value in [
                    ("does_bbo_price_change", 100.0, 105.0, False),
                    ("does_bbo_price_change", 99.0, 105.0, True),
                    ("does_bbo_price_change", 100.0, 106.0, True),
                    ("does_bbo_cross", 105.0, 110.0, False),
                    ("does_bbo_cross", 90.0, 100.0, False),
                    ("does_bbo_cross", 106.0, 110.0, True),
                    ("does_bbo_cross", 90.0, 99.0, True),
                ]
            ],
        ),
        _trace(
            "M8_all_getters_preserve_state",
            *[
                query(name, *args)
                for _ in range(3)
                for name, args in [
                    ("get_mid_price", ()),
                    ("get_bbo_spread", ()),
                    ("get_wmid_price", ()),
                    ("get_volume_weighted_mid_price", (20.0, True)),
                    ("get_price_impact", (16.0, True, True)),
                    ("get_size_for_price_impact_bps", (100.0, False, False)),
                ]
            ],
        ),
    ]
    cases.extend(
        [
            _trace(
                "D6_normalized_duplicate_tick_last_wins",
                delta(bids=[(100.2, 7.0, 7), (100.1, 12.0, 12)]),
            ),
            _trace(
                "D8_normalized_duplicate_final_zero_before_cross",
                delta(asks=[(99.2, 7.0, 7), (99.1, 0.0, 0)]),
            ),
            Scenario(
                "M3_asymmetric_volume_weighted_fixture",
                (
                    snapshot(
                        [(100.0, 1.0, 1), (110.0, 9.0, 2)],
                        [(99.0, 5.0, 3), (80.0, 5.0, 4)],
                    ),
                    query("get_volume_weighted_mid_price", 10.0, True, expected=99.25),
                    query("get_volume_weighted_mid_price", 0.5, True, expected=99.5),
                    query("get_volume_weighted_mid_price", 2.5, True, expected=102.5),
                    query(
                        "get_volume_weighted_mid_price", 247.5, False, expected=102.5
                    ),
                ),
            ),
            Scenario(
                "M4_zero_price_bid",
                (
                    snapshot([(2.0, 2.0, 2)], [(0.0, 1.0, 1)]),
                    query("get_volume_weighted_mid_price", 1.0, True, expected=1.0),
                    query(
                        "get_volume_weighted_mid_price",
                        1.5,
                        True,
                        expected=float("inf"),
                    ),
                ),
            ),
            Scenario(
                "M4_fractional_lot_exact_exhaustion",
                (
                    snapshot(
                        [(105.0, 0.3, 1), (106.0, 0.6, 2)],
                        [(100.0, 0.3, 3), (99.0, 0.6, 4)],
                    ),
                    query("get_volume_weighted_mid_price", 0.9, True),
                    query(
                        "get_volume_weighted_mid_price",
                        0.9001,
                        True,
                        expected=float("inf"),
                    ),
                    query("get_price_impact", 0.9, True, True, expected=1.0),
                    query("get_price_impact", 0.3001, True, True, expected=1.0),
                    query(
                        "get_price_impact", 0.9001, True, True, expected=float("inf")
                    ),
                    query("get_price_impact", 0.3001 * 105, True, False, expected=1.0),
                    query(
                        "get_price_impact",
                        0.9001 * 105,
                        True,
                        False,
                        expected=float("inf"),
                    ),
                ),
                lot=0.1,
            ),
        ]
    )
    cases.extend(
        [
            _trace(
                "S3_snapshot_normalized_tick_duplicates_then_delete",
                snapshot(
                    BASE_ASKS + [(105.2, 7.0, 9), (105.1, 12.0, 12)],
                    BASE_BIDS + [(100.2, 7.0, 9), (100.1, 12.0, 12)],
                ),
                delta([(105.8, 0.0, 0)], [(100.8, 0.0, 0)]),
            ),
            _trace(
                "S4_snapshot_normalized_tick_final_zero",
                snapshot(
                    BASE_ASKS + [(105.2, 7.0, 9), (105.1, 0.0, 0)],
                    BASE_BIDS + [(100.2, 7.0, 9), (100.1, 0.0, 0)],
                ),
            ),
            _trace(
                "S4_all_zero_snapshot_replaces_populated_book",
                snapshot([(105.0, 0.0, 0)], [(100.0, 0.0, 0)]),
                snapshot(),
            ),
            _trace(
                "D3_replace_last_levels_with_new_levels_same_batch",
                delta(
                    [(p, 0.0, 0) for p, _, _ in BASE_ASKS] + [(110.0, 17.0, 19)],
                    [(p, 0.0, 0) for p, _, _ in BASE_BIDS] + [(95.0, 18.0, 20)],
                ),
            ),
            _trace(
                "D3_delete_multiple_heads_preserves_remaining_metadata",
                delta(
                    [(105.0, 0.0, 0), (106.0, 0.0, 0)],
                    [(100.0, 0.0, 0), (99.0, 0.0, 0)],
                ),
            ),
            _trace(
                "M4_nonpositive_sizes_and_impact_bands",
                *[
                    query(name, *args, expected=expected)
                    for size in (0.0, -1.0)
                    for name, args, expected in [
                        ("get_volume_weighted_mid_price", (size, True), 102.0),
                        ("get_volume_weighted_mid_price", (size, False), 102.0),
                        ("get_price_impact", (size, True, True), 0.0),
                        ("get_price_impact", (size, False, False), 0.0),
                        ("get_size_for_price_impact_bps", (size, True, True), 0.0),
                        ("get_size_for_price_impact_bps", (size, False, False), 0.0),
                    ]
                ],
            ),
            _trace(
                "M6_full_depth_impact_band_base_and_quote",
                *[
                    query(
                        "get_size_for_price_impact_bps",
                        10000.0,
                        buy,
                        base,
                        expected=expected,
                    )
                    for buy, base, expected in [
                        (True, True, 120.0),
                        (False, True, 100.0),
                        (True, False, 12830.0),
                        (False, False, 9800.0),
                    ]
                ],
            ),
        ]
    )
    for orientation in range(1, 5):
        for hint in (orientation, 0):
            # Inputs themselves exercise each orientation. The normal adapter
            # also orders deltas consistently with the declared hint.
            reverse_asks = orientation in (2, 3)
            reverse_bids = orientation in (2, 4)
            cases.append(
                Scenario(
                    f"S5_orientation{orientation}_hint{hint}",
                    (
                        snapshot(
                            sorted(BASE_ASKS, reverse=reverse_asks),
                            sorted(BASE_BIDS, reverse=reverse_bids),
                        ),
                        delta(
                            [(106.0, 7.0, 9), (107.0, 8.0, 10)],
                            [(98.0, 9.0, 11), (99.0, 10.0, 12)],
                        ),
                    ),
                    sortedness=hint,
                    input_sortedness=orientation,
                )
            )
    for count in (1, 7, 8, 9, 64):
        ask_noops = (count - 1) // 2
        bid_noops = count - 1 - ask_noops
        cases.append(
            _trace(
                f"D12_compacted_total_batch_size_{count}",
                delta(
                    [(float(200 + j), 0.0, 0) for j in range(ask_noops)],
                    [(99.0, 8.0, 10)]
                    + [(float(20 + j), 0.0, 0) for j in range(bid_noops)],
                ),
            )
        )
    for count in (4, 7, 8, 9, 64):
        ask_noops = (count - 4) // 2
        bid_noops = count - 4 - ask_noops
        cases.append(
            _trace(
                f"C5_completed_batch_capacity_total_size_{count}",
                delta(
                    [(104.0, 7.0, 9), (106.0, 0.0, 0)]
                    + [(float(200 + j), 0.0, 0) for j in range(ask_noops)],
                    [(101.0, 8.0, 10), (99.0, 0.0, 0)]
                    + [(float(20 + j), 0.0, 0) for j in range(bid_noops)],
                ),
                capacity=4,
            )
        )
    for count in (7, 8, 9, 64):
        ask_noops = (count - 2) // 2
        bid_noops = count - 2 - ask_noops
        cases.append(
            _trace(
                f"C5_cross_cleanup_precedes_capacity_total_batch{count}",
                delta(
                    [(99.0, 8.0, 9)]
                    + [(float(200 + j), 0.0, 0) for j in range(ask_noops)],
                    [(101.0, 7.0, 10)]
                    + [(float(20 + j), 0.0, 0) for j in range(bid_noops)],
                ),
                capacity=4,
            )
        )
    for count in (7, 8):
        cases.append(
            _trace(
                f"D12_raw_large_compacts_to_total_{count}",
                delta(
                    [(106.0, 7.0, 7), (106.0, 0.0, 0), (106.0, 12.0, 12)],
                    [(99.0, 7.0, 7), (99.0, 0.0, 0), (99.0, 12.0, 12)]
                    + [(float(20 + j), 0.0, 0) for j in range(count - 2)],
                ),
            )
        )
    cases.extend(
        [
            _trace(
                "D1_absolute_ask_quantity_and_order_replacement",
                delta(asks=[(106.0, 7.0, 9)]),
            ),
            _trace(
                "D6_duplicate_ask_final_nonzero",
                delta(asks=[(106.0, 7.0, 7), (106.0, 0.0, 0), (106.0, 12.0, 12)]),
            ),
            _trace(
                "D6_duplicate_bid_final_zero",
                delta(bids=[(99.0, 7.0, 7), (99.0, 12.0, 12), (99.0, 0.0, 0)]),
            ),
            _trace(
                "D7_delete_ask_restore_across_calls",
                delta(asks=[(106.0, 0.0, 0)]),
                delta(asks=[(106.0, 7.0, 9)]),
            ),
            _trace(
                "D8_temporary_bid_cross_final_zero_preserves_asks",
                delta(bids=[(106.0, 7.0, 9), (106.0, 0.0, 0)]),
            ),
            _trace(
                "D9_both_sides_move_up_together",
                delta(
                    [(p, 0.0, 0) for p, _, _ in BASE_ASKS] + [(110.0, 8.0, 11)],
                    [(109.0, 7.0, 12)],
                ),
            ),
            _trace(
                "D10_partial_cross_prunes_only_crossed_asks",
                delta(bids=[(106.0, 8.0, 11)]),
            ),
            _trace(
                "M6_huge_finite_band_all_depth",
                *[
                    query(
                        "get_size_for_price_impact_bps",
                        1e300,
                        buy,
                        base,
                        expected=expected,
                    )
                    for buy, base, expected in [
                        (True, True, 120.0),
                        (False, True, 100.0),
                        (True, False, 12830.0),
                        (False, False, 9800.0),
                    ]
                ],
            ),
        ]
    )
    cases.extend(
        [
            _trace(
                "C8_deplete_bid_restore_delta",
                delta(bids=[(p, 0.0, 0) for p, _, _ in BASE_BIDS]),
                delta(bids=[(99.0, 7.0, 9)]),
            ),
            _trace(
                "C8_deplete_ask_restore_bbo",
                delta([(p, 0.0, 0) for p, _, _ in BASE_ASKS]),
                bbo((106.0, 7.0, 9), (99.0, 8.0, 10)),
            ),
            _trace(
                "C9_snapshot_recovery_after_ask_depletion",
                delta([(p, 0.0, 0) for p, _, _ in BASE_ASKS]),
                snapshot(),
            ),
            _trace(
                "B10_matching_zero_ask_refreshes_bid",
                bbo((105.0, 0.0, 0), (100.0, 7.0, 9)),
            ),
            _trace(
                "B10_matching_zero_bid_refreshes_ask",
                bbo((105.0, 7.0, 9), (100.0, 0.0, 0)),
            ),
            _trace(
                "B3_outward_ask_stationary_bid_refresh",
                bbo((106.0, 7.0, 9), (100.0, 8.0, 10)),
            ),
            _trace(
                "B3_outward_bid_stationary_ask_refresh",
                bbo((105.0, 7.0, 9), (99.0, 8.0, 10)),
            ),
            Scenario(
                "M6_fractional_sell_band_exact_below_above",
                (
                    snapshot([(0.11, 1.0, 1)], [(0.10, 1.0, 2), (0.09, 1.0, 3)]),
                    *[
                        query(
                            "get_size_for_price_impact_bps",
                            bps,
                            False,
                            base,
                            expected=expected,
                        )
                        for bps, base, expected in [
                            (999.999999, True, 1.0),
                            (1000.0, True, 2.0),
                            (1000.000001, True, 2.0),
                            (999.999999, False, 0.1),
                            (1000.0, False, 0.19),
                            (1000.000001, False, 0.19),
                        ]
                    ],
                ),
                tick=0.01,
            ),
        ]
    )
    for capacity in (4, 7):
        cycles = []
        for index in range(200):
            cycles += [
                delta(
                    [(104.0, float(index + 1), index + 1)],
                    [(101.0, float(index + 2), index + 2)],
                ),
                delta([(104.0, 0.0, 0)], [(101.0, 0.0, 0)]),
                delta(
                    [(float(104 + capacity), 45.0, 8)],
                    [(float(101 - capacity), 40.0, 4)],
                ),
            ]
        cases.append(
            Scenario(
                f"C6_ring_wrap_200_cycles_capacity{capacity}",
                (
                    snapshot(
                        [
                            (float(105 + j), float(15 + j * 10), j + 5)
                            for j in range(capacity)
                        ],
                        [
                            (float(100 - j), float(10 + j * 10), j + 1)
                            for j in range(capacity)
                        ],
                    ),
                    *cycles,
                ),
                capacity=capacity,
            )
        )
    return cases


def numeric_scenarios():
    cases = []
    for tick, lot in ((1.0, 1.0), (0.1, 0.01), (0.01, 0.001), (0.00001, 0.00000001)):

        def raw(price, size, orders):
            return (
                float(Decimal(price) * Decimal(str(tick))),
                float(Decimal(size) * Decimal(str(lot))),
                orders,
            )

        asks = [
            raw(p, q, n)
            for p, q, n in [(105, 15, 5), (106, 25, 6), (107, 35, 7), (108, 45, 8)]
        ]
        bids = [
            raw(p, q, n)
            for p, q, n in [(100, 10, 1), (99, 20, 2), (98, 30, 3), (97, 40, 4)]
        ]
        cases.append(
            Scenario(
                f"N1_grid_{tick}_{lot}",
                (
                    snapshot(asks, bids),
                    delta([raw(105, 1, 9)], [raw(100, 3, 10)]),
                    query(
                        "get_volume_weighted_mid_price",
                        float(Decimal(3) * Decimal(str(lot))),
                        True,
                    ),
                    query(
                        "get_price_impact",
                        float(Decimal(1) * Decimal(str(lot))),
                        True,
                        True,
                    ),
                ),
                tick=tick,
                lot=lot,
            )
        )
    cases.append(
        Scenario(
            "N2_80789_76_and_one_three_lots",
            (
                snapshot([(80789.77, 0.00000003, 3)], [(80789.76, 0.00000001, 1)]),
                delta([(80789.77, 0.00000001, 9)], [(80789.76, 0.00000003, 10)]),
            ),
            tick=0.01,
            lot=0.00000001,
        )
    )
    cases.append(
        Scenario(
            "N3_ticks_lots_and_orders_above_32_bits",
            (
                snapshot(
                    [(5000000005.0, 6000000000.0, 4294967301)],
                    [(5000000000.0, 7000000000.0, 4294967302)],
                ),
                query("get_wmid_price"),
                query("get_volume_weighted_mid_price", 3000000000.0, True),
            ),
            tick=1.0,
            lot=1.0,
        )
    )
    cases.extend(
        [
            Scenario(
                "N3_midpoint_sum_above_u64",
                (
                    snapshot([(1.2e19, 1.0, 1)], [(1e19, 1.0, 2)]),
                    query("get_mid_price", expected=1.1e19),
                    query("get_wmid_price", expected=1.1e19),
                    query("get_volume_weighted_mid_price", 1.0, True, expected=1.1e19),
                ),
            ),
            Scenario(
                "N3_aggregate_lots_above_u64",
                (
                    snapshot(
                        [(120.0, 1e19, 1), (122.0, 1e19, 2)],
                        [(100.0, 1e19, 3), (98.0, 1e19, 4)],
                    ),
                    query(
                        "get_size_for_price_impact_bps",
                        200.0,
                        True,
                        True,
                        expected=2e19,
                    ),
                    query(
                        "get_size_for_price_impact_bps",
                        200.0,
                        False,
                        True,
                        expected=2e19,
                    ),
                    query(
                        "get_size_for_price_impact_bps",
                        200.0,
                        True,
                        False,
                        expected=2.42e21,
                    ),
                    query(
                        "get_size_for_price_impact_bps",
                        200.0,
                        False,
                        False,
                        expected=1.98e21,
                    ),
                    query("get_volume_weighted_mid_price", 2e19, True, expected=110.0),
                    query("get_price_impact", 2e19, True, True, expected=2.0),
                    query("get_price_impact", 2e19, False, True, expected=2.0),
                ),
            ),
        ]
    )
    cases.extend(
        [
            Scenario(
                "N3_skewed_weighted_mid_exact_floor_ask_dominant",
                (
                    snapshot([(105.0, 1e16, 1)], [(100.0, 1.0, 2)]),
                    query("get_wmid_price", expected=104.0),
                ),
            ),
            Scenario(
                "N3_skewed_weighted_mid_exact_floor_bid_dominant",
                (
                    snapshot([(105.0, 1.0, 1)], [(100.0, 1e16, 2)]),
                    query("get_wmid_price", expected=100.0),
                ),
            ),
        ]
    )
    # Decimal-exact representable integers near the unsigned-64 price limit.
    # Mirrored extreme touch quantities expose a floating weighted mean that
    # rounds past either endpoint even when approximate equality would pass.
    for bid_price, ask_price in ((1e19, 1.2e19), (1.6e19, 1.8e19)):
        for ask_size, bid_size, endpoint in (
            (1e16, 1.0, ask_price),
            (1.0, 1e16, bid_price),
        ):
            cases.append(
                Scenario(
                    f"N3_skewed_weighted_mid_convexity_{bid_price}_{ask_price}_{ask_size}_{bid_size}",
                    (
                        snapshot(
                            [(ask_price, ask_size, 1)], [(bid_price, bid_size, 2)]
                        ),
                        query("get_wmid_price", expected=endpoint),
                    ),
                )
            )
    for field in (0, 1):
        for unsafe in (float(2**64), 1e308):
            row = [106.0, 7.0, 9]
            row[field] = unsafe
            for kind in ("snapshot", "deltas"):
                cases.append(
                    _trace(
                        f"N3_unsafe_scaled_u64_{kind}_field{field}_{unsafe}",
                        Event(
                            kind, (tuple(row),), tuple(BASE_BIDS), True, error="64-bit"
                        ),
                    )
                )
            ask = (unsafe, 7.0, 9) if field == 0 else (106.0, unsafe, 9)
            cases.append(
                _trace(
                    f"N3_unsafe_scaled_u64_bbo_field{field}_{unsafe}",
                    bbo(ask, (100.0, 8.0, 10), True, error="64-bit"),
                )
            )
    for field in (0, 1):
        for invalid in (-1.0, float("nan"), float("inf")):
            for position in (0, 1, 2):
                for kind in ("snapshot", "deltas"):
                    for side in ("asks", "bids"):
                        # All other rows form a valid, sorted, uncrossed payload.
                        # Require the invalid field diagnostic so unrelated
                        # crossing or sort rejection cannot satisfy the test.
                        rows = (
                            [(106.0, 7.0, 9), (107.0, 8.0, 10), (109.0, 9.0, 11)]
                            if side == "asks"
                            else [(100.0, 7.0, 9), (99.0, 8.0, 10), (98.0, 9.0, 11)]
                        )
                        bad = list(rows[position])
                        bad[field] = invalid
                        rows[position] = tuple(bad)
                        event = Event(
                            kind,
                            tuple(rows) if side == "asks" else tuple(BASE_ASKS),
                            tuple(BASE_BIDS) if side == "asks" else tuple(rows),
                            True,
                            error="invalid price" if field == 0 else "invalid size",
                        )
                        cases.append(
                            _trace(
                                f"N4_{kind}_{side}_field{field}_{invalid}_position{position}",
                                event,
                            )
                        )
    for orders in (-1, 2**64):
        for side in ("asks", "bids"):
            for kind in ("snapshot", "deltas"):
                asks = [(106.0, 7.0, orders if side == "asks" else 9)]
                bids = [(99.0, 8.0, orders if side == "bids" else 10)]
                cases.append(
                    _trace(
                        f"N4_norders_integer_domain_{kind}_{side}_{orders}",
                        Event(kind, tuple(asks), tuple(bids), True),
                    )
                )
            cases.append(
                _trace(
                    f"N4_norders_integer_domain_bbo_{side}_{orders}",
                    bbo(
                        (106.0, 7.0, orders if side == "asks" else 9),
                        (99.0, 8.0, orders if side == "bids" else 10),
                        True,
                    ),
                )
            )
    for bid, ask in ((105.0, 105.0), (106.0, 105.0)):
        cases.append(
            _trace(
                f"N5_invalid_bbo_{bid}_{ask}",
                bbo((ask, 8.0, 9), (bid, 7.0, 10), True, error="crossed bbo"),
            )
        )
    for field in (0, 1):
        for invalid in (-1.0, float("nan"), float("inf")):
            for side in ("ask", "bid"):
                ask, bid = [105.0, 8.0, 9], [100.0, 7.0, 10]
                (ask if side == "ask" else bid)[field] = invalid
                cases.append(
                    _trace(
                        f"N5_invalid_bbo_{side}_field{field}_{invalid}",
                        bbo(
                            ask,
                            bid,
                            True,
                            error="invalid price" if field == 0 else "invalid size",
                        ),
                    )
                )
    return cases


NORMAL_SEEDS = (7, 19, 41, 73, 101, 211, 307, 401, 509, 997)
EXTENDED_SEEDS = NORMAL_SEEDS + tuple(range(1001, 1031))


def generated_events(seed, actions=2000, capacity=4):
    """Generate raw events without reading a production book or reference state.

    Every 100-action cycle deliberately visits all operation families; random
    values vary quantities, metadata, batch widths and price pressure.
    """
    rng = Random(seed)
    anchor_floor = capacity + 100
    anchor = anchor_floor
    for step in range(actions):
        phase = step % 100

        def known_snapshot():
            return snapshot(
                [(float(anchor + 5 + j), float(15 + j), j + 5) for j in range(4)],
                [(float(anchor - j), float(10 + j), j + 1) for j in range(4)],
            )

        if phase in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24):
            forced = {
                10: ("known-state-snapshot", known_snapshot()),
                11: (
                    "completed-partial-cross",
                    delta(
                        [(float(anchor - 1), 8.0, 9)], [(float(anchor - 2), 7.0, 10)]
                    ),
                ),
                12: (
                    "temporary-cross-final-zero",
                    delta([(float(anchor - 2), 7.0, 7), (float(anchor - 2), 0.0, 0)]),
                ),
                13: ("known-state-snapshot", known_snapshot()),
                14: ("empty-delta", delta()),
                15: (
                    "one-side-depletion",
                    delta([(float(anchor + 5 + j), 0.0, 0) for j in range(4)]),
                ),
                16: (
                    "update-while-one-side-empty",
                    delta(bids=[(float(anchor + 1), 7.0, 9)]),
                ),
                17: ("delta-side-recovery", delta(asks=[(float(anchor + 5), 8.0, 10)])),
                18: (
                    "bbo-multiple-level-outward",
                    bbo((float(anchor + 7), 9.0, 11), (float(anchor - 2), 10.0, 12)),
                ),
                19: (
                    "duplicates-both-sides",
                    delta(
                        [
                            (float(anchor + 7), 7.0, 7),
                            (float(anchor + 7), 12.0, 12),
                            (float(anchor + 7), 0.0, 0),
                        ],
                        [
                            (float(anchor - 2), 7.0, 7),
                            (float(anchor - 2), 0.0, 0),
                            (float(anchor - 2), 12.0, 12),
                        ],
                    ),
                ),
                21: ("known-state-snapshot", known_snapshot()),
                22: (
                    "reject-crossed-snapshot",
                    snapshot(
                        [(float(anchor - 1), 7.0, 9)], [(float(anchor), 8.0, 10)], True
                    ),
                ),
                23: (
                    "reject-invalid-delta",
                    Event(
                        "deltas",
                        ((float(anchor + 6), 7.0, 9), (float(anchor + 7), -1.0, 10)),
                        (),
                        True,
                        error="invalid size",
                    ),
                ),
                24: (
                    "reject-locked-bbo",
                    bbo(
                        (float(anchor + 2), 7.0, 9), (float(anchor + 2), 8.0, 10), True
                    ),
                ),
            }
            family, event = forced[phase]
            yield replace(event, name=family)
            continue
        if phase in (0, 1):
            if phase == 0 and step:
                yield Event("clear")
            else:
                depth = min(capacity + 3, 1030)
                asks = [
                    (
                        float(anchor + 5 + j),
                        float(rng.randint(1, 100)),
                        rng.randint(1, 1000),
                    )
                    for j in range(depth)
                ]
                bids = [
                    (
                        float(anchor - j),
                        float(rng.randint(1, 100)),
                        rng.randint(1, 1000),
                    )
                    for j in range(depth)
                ]
                yield snapshot(asks, bids)
        elif phase in (20, 40, 60, 80):
            anchor = rng.randint(anchor_floor, anchor_floor + 200)
            yield bbo(
                (float(anchor + 5), float(rng.randint(1, 30)), rng.randint(1, 1000)),
                (float(anchor), float(rng.randint(1, 30)), rng.randint(1, 1000)),
            )
        elif phase == 99:
            anchor = rng.randint(anchor_floor, anchor_floor + 200)
            yield snapshot(
                [
                    (float(anchor + 5 + j), float(j + 1), j + 1)
                    for j in range(capacity + 2)
                ],
                [(float(anchor - j), float(j + 2), j + 2) for j in range(capacity + 2)],
            )
        else:
            count = (1, 7, 8, 9, 64)[rng.randrange(5)] if phase % 3 == 0 else 1
            asks, bids = [], []
            for j in range(count):
                price = float(anchor + rng.randint(5, 20))
                bid_price = float(anchor - rng.randint(0, 15))
                quantity = float(rng.choice((0, 0, 1, 3, 7, 20)))
                row = (price, quantity, rng.randint(1, 1000))
                bid_row = (
                    bid_price,
                    float(rng.choice((0, 1, 3, 12))),
                    rng.randint(1, 1000),
                )
                if phase % 5 != 0:
                    asks.append(row)
                if phase % 7 != 0:
                    bids.append(bid_row)
            if phase % 11 == 0:
                # Explicit duplicate final-write patterns, including final zero.
                asks.extend(
                    [
                        (float(anchor + 7), 7.0, 7),
                        (float(anchor + 7), 0.0, 0),
                        (float(anchor + 7), 12.0 if phase % 2 else 0.0, 12),
                    ]
                )
            yield delta(asks, bids)

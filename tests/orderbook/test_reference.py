"""Oracle hand calculations prevent shared implementation mistakes going green."""

from fractions import Fraction
import pytest

from .reference import ReferenceBook
from .scenarios import BASE_ASKS, BASE_BIDS


def book(capacity=None, tick=1.0, lot=1.0):
    result = ReferenceBook(tick, lot, capacity)
    result.snapshot(BASE_ASKS, BASE_BIDS)
    return result


def test_reference_decimal_conversion_and_large_integer_arithmetic():
    reference = ReferenceBook(0.01, 0.00000001)
    reference.snapshot(
        [(80789.77, 0.00000003, 4294967301)], [(80789.76, 0.00000001, 4294967302)]
    )
    assert reference.integer_rows("bids") == [(8078976, 1, 4294967302)]
    assert reference.integer_rows("asks") == [(8078977, 3, 4294967301)]
    assert reference.query("get_wmid_price") == 80789.76


def test_reference_hand_calculated_base_metrics():
    reference = book()
    assert reference.query("get_mid_price") == 102
    assert reference.query("get_bbo_spread") == 5
    assert reference.query("get_wmid_price") == 103
    assert reference.query("get_volume_weighted_mid_price", 20.0, True) == 102.375
    assert reference.query("get_price_impact", 15.0, True, True) == 0
    assert reference.query("get_price_impact", 16.0, True, True) == 1
    assert reference.query("get_price_impact", 11.0, False, True) == 1
    assert reference.query("get_price_impact", 100.0, False, True) == 3
    assert reference.query("get_price_impact", 101.0, False, True) == float("inf")


def test_reference_duplicates_are_last_absolute_update_before_crossing():
    reference = book()
    reference.deltas(
        [(99.2, 7.0, 7), (99.1, 0.0, 0)],
        [(99.0, 7.0, 7), (99.0, 0.0, 0), (99.0, 12.0, 12)],
    )
    assert reference.integer_rows("asks") == [
        (105, 15, 5),
        (106, 25, 6),
        (107, 35, 7),
        (108, 45, 8),
    ]
    assert reference.integer_rows("bids") == [
        (100, 10, 1),
        (99, 12, 12),
        (98, 30, 3),
        (97, 40, 4),
    ]


def test_reference_capacity_applies_to_completed_batch():
    reference = book(capacity=4)
    reference.deltas(
        [(104.0, 7.0, 9), (106.0, 0.0, 0)], [(101.0, 8.0, 10), (99.0, 0.0, 0)]
    )
    assert reference.integer_rows("asks") == [
        (104, 7, 9),
        (105, 15, 5),
        (107, 35, 7),
        (108, 45, 8),
    ]
    assert reference.integer_rows("bids") == [
        (101, 8, 10),
        (100, 10, 1),
        (98, 30, 3),
        (97, 40, 4),
    ]


def test_reference_partial_cross_and_complete_wipe_contract():
    reference = book()
    reference.deltas([(96.0, 8.0, 9)], [])
    assert reference.integer_rows("bids") == [
        (100, 10, 1),
        (99, 20, 2),
        (98, 30, 3),
        (97, 40, 4),
    ]
    reference.deltas([(99.0, 8.0, 9)], [])
    assert reference.integer_rows("bids") == [(98, 30, 3), (97, 40, 4)]


def test_reference_bbo_prunes_and_only_explicit_updates_restore():
    reference = book()
    reference.bbo((107.0, 7.0, 9), (98.0, 8.0, 10))
    assert reference.integer_rows("asks") == [(107, 7, 9), (108, 45, 8)]
    assert reference.integer_rows("bids") == [(98, 8, 10), (97, 40, 4)]
    reference.deltas([(106.0, 2.0, 11)], [(99.0, 3.0, 12)])
    assert reference.integer_rows("bids") == [(99, 3, 12), (98, 8, 10), (97, 40, 4)]


def test_reference_atomic_invalid_snapshot_delta_and_bbo():
    reference = book()
    before = reference.rows("asks"), reference.rows("bids")
    for operation, args in [
        (reference.snapshot, ([(99.0, 1.0, 1)], BASE_BIDS)),
        (reference.deltas, ([(106.0, 2.0, 1), (107.0, float("nan"), 1)], [])),
        (reference.bbo, ((106.0, 0.0, 0), (100.0, 7.0, 1))),
    ]:
        with pytest.raises(ValueError):
            operation(*args)
        assert (reference.rows("asks"), reference.rows("bids")) == before


def test_reference_exact_rational_exhaustion_and_impact_bands():
    reference = ReferenceBook(1.0, 0.1)
    reference.snapshot(
        [(200.0, 0.3, 1), (202.0, 0.6, 2)], [(100.0, 0.3, 3), (99.0, 0.6, 4)]
    )
    assert reference.query("get_volume_weighted_mid_price", 0.9, True) == float(
        Fraction(451, 3)
    )
    assert reference.query("get_volume_weighted_mid_price", 0.9001, True) == float(
        "inf"
    )
    assert reference.query("get_size_for_price_impact_bps", 100.0, True, True) == 0.9
    assert reference.query("get_size_for_price_impact_bps", 99.999, True, True) == 0.3
    assert reference.query("get_size_for_price_impact_bps", 100.0, False, False) == 89.4
    assert reference.query("does_bbo_cross", 200.0, 300.0) is False
    assert reference.query("does_bbo_cross", 201.0, 300.0) is True


def test_reference_unsigned_64_bit_normalization_bound():
    reference = ReferenceBook()
    reference.snapshot([(120.0, 1e19, 1)], [(100.0, 1e19, 2)])
    assert reference.integer_rows("asks") == [(120, 10000000000000000000, 1)]
    before = reference.rows("asks"), reference.rows("bids")
    for row in [(float(2**64), 1.0, 1), (120.0, float(2**64), 1), (120.0, 1e308, 1)]:
        with pytest.raises(ValueError, match="64-bit"):
            reference.deltas([row], [])
        assert (reference.rows("asks"), reference.rows("bids")) == before

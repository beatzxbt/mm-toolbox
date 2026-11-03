"""One scenario suite against exactly three implementations and one oracle.

Run ordinary coverage with ``pytest tests/orderbook/test_scenarios.py``.
Use ``--orderbook-extended`` for 40 fixed seeds x 5000 events, including cap1024.
Every assertion includes a replayable seed, step, raw event and complete state.
"""

from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN
from math import isinf
from functools import lru_cache

import pytest

from .adapters import TARGETS, make_target
from .reference import ReferenceBook
from .scenarios import (
    EXTENDED_SEEDS,
    NORMAL_SEEDS,
    generated_events,
    numeric_scenarios,
    scenarios,
)


@lru_cache(maxsize=262144)
def _export_units(value, quantum):
    """Floor raw exports, correcting only representation noise near an integer.

    This test-owned conversion does not import production precision helpers.
    A tolerance of one millionth of a unit cannot conceal a one-tick/lot error.
    Off-grid standard levels intentionally retain their original raw values.
    """
    ratio = Decimal(str(value)) / quantum
    nearest = ratio.to_integral_value(rounding=ROUND_HALF_EVEN)
    if abs(ratio - nearest) < Decimal("0.000001"):
        return int(nearest)
    return int(ratio.to_integral_value(rounding=ROUND_FLOOR))


def _assert_value(actual, expected, context):
    if isinstance(expected, bool):
        assert bool(actual) is expected, (
            f"{context}\nexpected={expected!r}\nactual={actual!r}"
        )
    elif isinf(expected):
        assert actual == float("inf"), (
            f"{context}\nexpected=positive infinity\nactual={actual!r}"
        )
    else:
        assert actual == pytest.approx(expected, rel=2e-13, abs=1e-14), (
            f"{context}\nexpected={expected!r}\nactual={actual!r}"
        )


def assert_state(target, reference, context):
    raw = {side: target.rows(side) for side in ("asks", "bids")}
    for side, rows in raw.items():
        actual = [
            (_export_units(p, reference.tick), _export_units(q, reference.lot), n)
            for p, q, n in rows
        ]
        expected = reference.integer_rows(side)
        assert actual == expected, (
            f"{context}\nside={side}\nexpected={expected!r}"
            f"\nactual={actual!r}\nraw_actual={rows!r}"
        )
    # Full-row equality includes side counts and touch quantity/order metadata.
    # Query the real BBO getter as well, independently of full-depth exports.
    if reference.asks and reference.bids:
        touch = target.query("get_bbo")
        expected_touch = (
            reference.integer_rows("bids")[0],
            reference.integer_rows("asks")[0],
        )
        actual_touch = tuple(
            (_export_units(p, reference.tick), _export_units(q, reference.lot), n)
            for p, q, n in touch
        )
        assert actual_touch == expected_touch, (
            f"{context}\nexpected BBO={expected_touch!r}\nactual BBO={actual_touch!r}"
        )
        assert target.query("counts") == (len(reference.bids), len(reference.asks)), (
            context
        )
        for name in ("get_mid_price", "get_bbo_spread", "get_wmid_price"):
            actual = target.query(name)
            _assert_value(actual, reference.query(name), f"{context}\nquery={name}")
            if name == "get_wmid_price":
                # A rounded weighted mean must stay inside the normalized touch.
                # Keep this strict: approximate equality at very large prices
                # could otherwise conceal an overshoot by several float ulps.
                bid_bound = expected_touch[0][0] * float(reference.tick)
                ask_bound = expected_touch[1][0] * float(reference.tick)
                assert bid_bound <= actual <= ask_bound, (
                    f"{context}\nquery={name}\nexpected bounds=[{bid_bound!r}, {ask_bound!r}]"
                    f"\nactual={actual!r}"
                )
    else:
        for name in (
            "get_mid_price",
            "get_bbo_spread",
            "get_wmid_price",
            "get_volume_weighted_mid_price",
            "get_price_impact",
            "get_size_for_price_impact_bps",
            "does_bbo_price_change",
            "does_bbo_cross",
            "get_bbo",
        ):
            args = {
                "get_volume_weighted_mid_price": (1.0, True),
                "get_price_impact": (1.0, True, True),
                "get_size_for_price_impact_bps": (1.0, True, True),
                "does_bbo_price_change": (1.0, 2.0),
                "does_bbo_cross": (1.0, 2.0),
            }.get(name, ())
            try:
                target.query(name, *args)
            except (ValueError, RuntimeError):
                pass
            else:
                pytest.fail(f"{context}\nquery={name} must reject incomplete book")


def replay(
    target_name,
    events,
    *,
    tick=1.0,
    lot=1.0,
    capacity=64,
    sortedness=0,
    input_sortedness=None,
    seed="scenario",
):
    target = make_target(target_name, tick, lot, capacity, sortedness)
    if input_sortedness is not None:
        target.sortedness = input_sortedness
    reference = ReferenceBook(
        tick, lot, None if target_name == "standard" else capacity
    )
    for step, event in enumerate(events):
        context = f"target={target_name} seed={seed} capacity={capacity} step={step} event={event!r}"
        if event.kind == "query":
            before = (reference.integer_rows("asks"), reference.integer_rows("bids"))
            expected = reference.query(event.name, *event.args)
            if event.expected is not None:
                _assert_value(expected, event.expected, f"oracle checkpoint: {context}")
            _assert_value(target.query(event.name, *event.args), expected, context)
            assert before == (
                reference.integer_rows("asks"),
                reference.integer_rows("bids"),
            ), context
        else:
            args = () if event.kind == "clear" else (event.asks, event.bids)
            if event.reject:
                before = (
                    reference.integer_rows("asks"),
                    reference.integer_rows("bids"),
                )
                with pytest.raises(ValueError):
                    getattr(reference, event.kind)(*args)
                try:
                    getattr(target, event.kind)(*args)
                except (ValueError, RuntimeError, OverflowError) as exc:
                    if event.error:
                        assert event.error in str(exc).lower(), (
                            f"{context}\nexpected rejection reason={event.error!r}\nactual={exc!r}"
                        )
                else:
                    pytest.fail(
                        f"{context}\nexpected=atomic rejection\nactual=accepted"
                    )
                assert before == (
                    reference.integer_rows("asks"),
                    reference.integer_rows("bids"),
                ), context
            else:
                try:
                    getattr(reference, event.kind)(*args)
                except Exception as exc:
                    pytest.fail(
                        f"{context}\nexpected=valid generated/reference event\nactual=oracle raised {exc!r}"
                    )
                try:
                    getattr(target, event.kind)(*args)
                except Exception as exc:
                    pytest.fail(
                        f"{context}\nexpected={reference.rows('asks'), reference.rows('bids')}\nactual=raised {exc!r}"
                    )
        assert_state(target, reference, context)
        if (
            isinstance(seed, int)
            and step % 37 == 0
            and reference.asks
            and reference.bids
        ):
            # Read-only depth calculations at deterministic random-run checkpoints.
            # The requests are derived only from the independent integer state.
            quantity = float(
                min(
                    sum(q for q, _ in reference.asks.values()),
                    sum(q for q, _ in reference.bids.values()),
                )
                * reference.lot
            )
            for name, args in [
                ("get_volume_weighted_mid_price", (quantity, True)),
                (
                    "get_volume_weighted_mid_price",
                    (quantity + float(reference.lot), True),
                ),
                ("get_price_impact", (quantity, True, True)),
                ("get_price_impact", (quantity, False, True)),
                ("get_size_for_price_impact_bps", (100.0, True, True)),
                ("get_size_for_price_impact_bps", (100.0, False, False)),
            ]:
                _assert_value(
                    target.query(name, *args),
                    reference.query(name, *args),
                    f"{context}\ncheckpoint query={name} args={args}",
                )
            assert_state(target, reference, context + " after calculation checkpoint")


ALL_SCENARIOS = scenarios() + numeric_scenarios()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda scenario: scenario.id)
def test_orderbook_scenario(target, scenario):
    replay(
        target,
        scenario.events,
        tick=scenario.tick,
        lot=scenario.lot,
        capacity=scenario.capacity,
        sortedness=scenario.sortedness,
        input_sortedness=scenario.input_sortedness,
        seed=scenario.id,
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "seed_index,seed", list(enumerate(NORMAL_SEEDS)), ids=lambda value: str(value)
)
def test_seeded_orderbook_replay(target, seed_index, seed):
    capacity = (4, 7, 64)[seed_index % 3]
    replay(target, generated_events(seed, 2000, capacity), capacity=capacity, seed=seed)


@pytest.mark.orderbook_extended
def test_extended_orderbook_replay():
    # Shared conftest visibly deselects this optional larger run unless requested.
    for index, seed in enumerate(EXTENDED_SEEDS):
        capacity = (4, 7, 64, 1024)[index % 4]
        events = tuple(generated_events(seed, 5000, capacity))
        for target in TARGETS:
            replay(target, events, capacity=capacity, seed=seed)
        print(
            f"extended seed={seed} capacity={capacity} actions=5000 targets=3 complete",
            flush=True,
        )


def _raw_ingest(target, target_name, kind, asks, bids):
    """Bypass adapter ordering for sort-inference failure atomicity checks."""
    if target_name == "advanced-cython":
        getattr(target.book, kind)(asks, bids)
    else:
        from mm_toolbox.orderbook.advanced import OrderbookLevels

        def levels(rows):
            return OrderbookLevels.from_list(
                [row[0] for row in rows],
                [row[1] for row in rows],
                [row[2] for row in rows],
            )

        getattr(target.book, "consume_" + kind)(levels(asks), levels(bids))


@pytest.mark.parametrize("target_name", TARGETS[1:])
@pytest.mark.parametrize(
    "failure",
    (
        "crossed-first-snapshot",
        "unsorted-first-snapshot",
        "unsorted-first-deltas",
        "unsorted-asks-first-snapshot",
        "unsorted-asks-first-deltas",
    ),
)
def test_rejected_first_order_inference_is_atomic(target_name, failure):
    target = make_target(target_name)
    reference = ReferenceBook(capacity=64)
    if failure in ("unsorted-first-deltas", "unsorted-asks-first-deltas"):
        target.snapshot([(105.0, 1.0, 1)], [(100.0, 1.0, 2)])
        reference.snapshot([(105.0, 1.0, 1)], [(100.0, 1.0, 2)])
        kind = "deltas"
        asks, bids = (
            [(108.0, 2.0, 3), (107.0, 3.0, 4)],
            [(99.0, 4.0, 5), (97.0, 5.0, 6), (98.0, 6.0, 7)],
        )
        reason = "unsorted"
    elif failure in ("unsorted-first-snapshot", "unsorted-asks-first-snapshot"):
        kind = "snapshot"
        asks, bids = (
            [(106.0, 2.0, 3), (105.0, 3.0, 4)],
            [(99.0, 4.0, 5), (97.0, 5.0, 6), (98.0, 6.0, 7)],
        )
        reason = "unsorted"
    else:
        kind = "snapshot"
        asks, bids = (
            [(105.0, 2.0, 3), (104.0, 3.0, 4)],
            [(106.0, 4.0, 5), (103.0, 5.0, 6)],
        )
        reason = "crossed"
    if failure.startswith("unsorted-asks"):
        asks = [(108.0, 2.0, 3), (106.0, 3.0, 4), (107.0, 4.0, 5)]
        bids = [(98.0, 5.0, 6), (99.0, 6.0, 7)]
    context = f"target={target_name} seed={failure} step=0 event={kind, asks, bids}"
    with pytest.raises(ValueError, match="(?i)" + reason):
        _raw_ingest(target, target_name, kind, asks, bids)
    assert_state(target, reference, context)
    valid_asks, valid_bids = (
        [(106.0, 7.0, 8), (107.0, 8.0, 9)],
        [(99.0, 9.0, 10), (98.0, 10.0, 11)],
    )
    if failure == "crossed-first-snapshot":
        valid_bids = list(reversed(valid_bids))
    _raw_ingest(target, target_name, kind, valid_asks, valid_bids)
    getattr(reference, kind)(valid_asks, valid_bids)
    assert_state(
        target, reference, context + " step=1 opposite valid order after rejection"
    )


@pytest.mark.parametrize("target_name", TARGETS)
def test_all_getters_leave_full_depth_unchanged(target_name):
    from .scenarios import BASE_ASKS, BASE_BIDS

    target = make_target(target_name)
    target.snapshot(BASE_ASKS, BASE_BIDS)
    before = (target.rows("asks"), target.rows("bids"))
    for _ in range(5):
        target.rows("asks")
        target.rows("bids")
        for name, args in [
            ("get_bbo", ()),
            ("counts", ()),
            ("get_mid_price", ()),
            ("get_bbo_spread", ()),
            ("get_wmid_price", ()),
            ("get_volume_weighted_mid_price", (20.0, True)),
            ("get_price_impact", (16.0, True, True)),
            ("get_size_for_price_impact_bps", (100.0, False, False)),
            ("does_bbo_price_change", (100.0, 105.0)),
            ("does_bbo_cross", (105.0, 110.0)),
        ]:
            target.query(name, *args)
    assert (target.rows("asks"), target.rows("bids")) == before, (
        f"target={target_name} seed=getter-stability step=5 event=repeat all getters\n"
        f"expected={before!r}\nactual={target.rows('asks'), target.rows('bids')}"
    )


@pytest.mark.parametrize("index,seed", list(enumerate(NORMAL_SEEDS)))
def test_seeded_corpus_contains_required_event_families(index, seed):
    from collections import Counter

    events = tuple(generated_events(seed, 2000, (4, 7, 64)[index % 3]))
    families = Counter(event.name for event in events)
    required = {
        "known-state-snapshot",
        "completed-partial-cross",
        "temporary-cross-final-zero",
        "empty-delta",
        "one-side-depletion",
        "update-while-one-side-empty",
        "delta-side-recovery",
        "bbo-multiple-level-outward",
        "duplicates-both-sides",
        "reject-crossed-snapshot",
        "reject-invalid-delta",
        "reject-locked-bbo",
    }
    context = (
        f"seed={seed} step=corpus event=required-family-audit actual={dict(families)!r}"
    )
    assert len(events) == 2000, context
    assert required <= families.keys(), context
    assert all(families[name] == 20 for name in required - {"known-state-snapshot"}), (
        context
    )
    assert all(
        kind in {event.kind for event in events}
        for kind in ("snapshot", "deltas", "bbo", "clear")
    ), context
    assert any(
        event.kind == "deltas" and len(event.asks) + len(event.bids) == 1
        for event in events
    ), context
    assert any(
        event.kind == "deltas" and len(event.asks) + len(event.bids) > 8
        for event in events
    ), context

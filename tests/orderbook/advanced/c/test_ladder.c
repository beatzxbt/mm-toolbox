/**
 * @file test_ladder.c
 * @brief µnit tests for ring-backed orderbook_ladder.c.
 */

#include "munit.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_ladder.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_types.h"

#define TICK_SIZE 0.01
#define LOT_SIZE 0.001
#define TICK_SIZE_RECIP (1.0 / TICK_SIZE)
#define LOT_SIZE_RECIP (1.0 / LOT_SIZE)
#define PRICE_TO_TICKS(price) price_to_tick((price), TICK_SIZE_RECIP)
#define SIZE_TO_LOTS(size) size_to_lot((size), LOT_SIZE_RECIP)

static OrderbookEntry make_entry(double price, double size, uint64_t norders) {
    OrderbookEntry entry = {
        .ticks = PRICE_TO_TICKS(price),
        .lots = SIZE_TO_LOTS(size),
        .norders = norders,
        ._pad = 0
    };
    return entry;
}

/**
 * @brief Insert a one-lot fixture entry at a logical ladder index.
 *
 * @param ladder Ladder under test.
 * @param index Logical index to populate.
 * @param price Price to convert into ticks.
 */
static void set_entry(OrderbookLadderData* ladder, uint64_t index, double price) {
    OrderbookEntry entry = make_entry(price, 1.0, 1);
    c_ladder_insert_entry(ladder, index, &entry);
}

/** @test Rolling right at the start preserves sorted logical order and wraps head. */
static MunitResult test_roll_right_at_start(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 2,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);

    c_ladder_roll_right(&ladder, 0);
    OrderbookEntry new_entry = make_entry(99.0, 1.0, 1);
    c_ladder_insert_entry(&ladder, 0, &new_entry);
    ladder.num_levels = 3;

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(99.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(101.0));
    munit_assert_uint64(ladder.head, ==, 4);
    return MUNIT_OK;
}

/** @test Rolling right in the middle shifts the suffix while preserving earlier levels. */
static MunitResult test_roll_right_in_middle(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    set_entry(&ladder, 2, 102.0);

    c_ladder_roll_right(&ladder, 1);
    OrderbookEntry new_entry = make_entry(100.5, 2.0, 1);
    c_ladder_insert_entry(&ladder, 1, &new_entry);
    ladder.num_levels = 4;

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->lots, ==, SIZE_TO_LOTS(2.0));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(101.0));
    munit_assert_uint64(c_ladder_at(&ladder, 3)->ticks, ==, PRICE_TO_TICKS(102.0));
    munit_assert_uint64(ladder.head, ==, 4);
    return MUNIT_OK;
}

/** @test Rolling right near a wrapped head uses the shorter prefix shift path. */
static MunitResult test_roll_right_uses_prefix_shift(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[8];
    OrderbookLadderData ladder = {
        .num_levels = 5,
        .max_levels = 8,
        .levels = levels,
        .head = 6,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 100.01);
    set_entry(&ladder, 2, 100.02);
    set_entry(&ladder, 3, 100.03);
    set_entry(&ladder, 4, 100.04);

    c_ladder_roll_right(&ladder, 2);
    OrderbookEntry new_entry = make_entry(150.0, 2.0, 1);
    c_ladder_insert_entry(&ladder, 2, &new_entry);
    ladder.num_levels = 6;

    munit_assert_uint64(ladder.head, ==, 5);
    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.01));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->lots, ==, SIZE_TO_LOTS(2.0));
    munit_assert_uint64(c_ladder_at(&ladder, 3)->ticks, ==, PRICE_TO_TICKS(100.02));
    munit_assert_uint64(c_ladder_at(&ladder, 4)->ticks, ==, PRICE_TO_TICKS(100.03));
    munit_assert_uint64(c_ladder_at(&ladder, 5)->ticks, ==, PRICE_TO_TICKS(100.04));
    return MUNIT_OK;
}

/** @test Rolling right at max capacity drops the tail instead of writing past capacity. */
static MunitResult test_roll_right_at_max_capacity(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[3];
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 3,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 100.01);
    set_entry(&ladder, 2, 100.02);

    c_ladder_roll_right(&ladder, 0);
    OrderbookEntry new_entry = make_entry(99.0, 2.0, 1);
    c_ladder_insert_entry(&ladder, 0, &new_entry);

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(99.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(100.01));
    return MUNIT_OK;
}

/** @test orderbook_entry_assign sets _pad to 0. */
static MunitResult test_orderbook_entry_assign_pad_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry entry = {0};
    orderbook_entry_assign(&entry, 10000, 1500, 5);
    munit_assert_uint64(entry.ticks, ==, 10000);
    munit_assert_uint64(entry.lots, ==, 1500);
    munit_assert_uint64(entry.norders, ==, 5);
    munit_assert_uint64(entry._pad, ==, 0);
    return MUNIT_OK;
}

/** @test Rolling right beyond the populated count leaves ladder contents unchanged. */
static MunitResult test_roll_right_beyond_count(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 2,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    c_ladder_roll_right(&ladder, 5);

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(101.0));
    return MUNIT_OK;
}

/** @test Rolling left at the start removes the top logical level and advances contents. */
static MunitResult test_roll_left_at_start(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    set_entry(&ladder, 2, 102.0);

    c_ladder_roll_left(&ladder, 0);
    ladder.num_levels = 2;

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(101.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(102.0));
    munit_assert_uint64(ladder.head, ==, 1);
    return MUNIT_OK;
}

/** @test Rolling left in the middle removes only the target logical entry. */
static MunitResult test_roll_left_in_middle(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    set_entry(&ladder, 2, 102.0);

    c_ladder_roll_left(&ladder, 1);
    ladder.num_levels = 2;

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(102.0));
    return MUNIT_OK;
}

/** @test Rolling left near a wrapped head uses the shorter prefix shift path. */
static MunitResult test_roll_left_uses_prefix_shift(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[8];
    OrderbookLadderData ladder = {
        .num_levels = 5,
        .max_levels = 8,
        .levels = levels,
        .head = 6,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 100.01);
    set_entry(&ladder, 2, 100.02);
    set_entry(&ladder, 3, 100.03);
    set_entry(&ladder, 4, 100.04);

    c_ladder_roll_left(&ladder, 1);
    ladder.num_levels = 4;

    munit_assert_uint64(ladder.head, ==, 7);
    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.02));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(100.03));
    munit_assert_uint64(c_ladder_at(&ladder, 3)->ticks, ==, PRICE_TO_TICKS(100.04));
    return MUNIT_OK;
}

/** @test Rolling left at the end drops the tail while preserving earlier entries. */
static MunitResult test_roll_left_at_end(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    set_entry(&ladder, 2, 102.0);

    c_ladder_roll_left(&ladder, 2);
    ladder.num_levels = 2;

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(101.0));
    return MUNIT_OK;
}

/** @test Rolling left beyond the populated count leaves ladder contents unchanged. */
static MunitResult test_roll_left_beyond_count(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 2,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    c_ladder_roll_left(&ladder, 5);

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(101.0));
    return MUNIT_OK;
}

/** @test Inserting an entry copies ticks, lots, and order count into storage. */
static MunitResult test_insert_entry(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookLadderData ladder = {
        .num_levels = 0,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };
    OrderbookEntry new_entry = make_entry(100.5, 2.0, 3);

    c_ladder_insert_entry(&ladder, 0, &new_entry);

    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.5));
    munit_assert_uint64(c_ladder_at(&ladder, 0)->lots, ==, SIZE_TO_LOTS(2.0));
    munit_assert_uint64(c_ladder_at(&ladder, 0)->norders, ==, 3);
    return MUNIT_OK;
}

/** @test Top and bottom accessors resolve logical ends when the head is wrapped. */
static MunitResult test_top_and_bottom_wrapped_head(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[6];
    OrderbookLadderData ladder = {
        .num_levels = 4,
        .max_levels = 6,
        .levels = levels,
        .head = 4,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.00);
    set_entry(&ladder, 1, 100.01);
    set_entry(&ladder, 2, 100.02);
    set_entry(&ladder, 3, 100.03);

    munit_assert_uint64(c_ladder_top(&ladder)->ticks, ==, PRICE_TO_TICKS(100.00));
    munit_assert_uint64(c_ladder_bottom(&ladder)->ticks, ==, PRICE_TO_TICKS(100.03));
    return MUNIT_OK;
}

/** @test Exporting wrapped ladder data preserves logical order and public fields. */
static MunitResult test_export_levels_wrapped_head(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[6];
    OrderbookLevel exported[4];
    OrderbookEntry entry;
    OrderbookLadderData ladder = {
        .num_levels = 4,
        .max_levels = 6,
        .levels = levels,
        .head = 4,
        .is_price_ascending = 1
    };

    entry = make_entry(100.00, 1.0, 1);
    c_ladder_insert_entry(&ladder, 0, &entry);
    entry = make_entry(100.01, 2.0, 2);
    c_ladder_insert_entry(&ladder, 1, &entry);
    entry = make_entry(100.02, 3.0, 3);
    c_ladder_insert_entry(&ladder, 2, &entry);
    entry = make_entry(100.03, 4.0, 4);
    c_ladder_insert_entry(&ladder, 3, &entry);

    c_ladder_export_levels(exported, &ladder, TICK_SIZE, LOT_SIZE);

    munit_assert_double_equal(exported[0].price, 100.00, 10);
    munit_assert_double_equal(exported[1].price, 100.01, 10);
    munit_assert_double_equal(exported[2].price, 100.02, 10);
    munit_assert_double_equal(exported[3].price, 100.03, 10);
    munit_assert_double_equal(exported[0].size, 1.0, 10);
    munit_assert_double_equal(exported[1].size, 2.0, 10);
    munit_assert_double_equal(exported[2].size, 3.0, 10);
    munit_assert_double_equal(exported[3].size, 4.0, 10);
    munit_assert_uint64(exported[0].norders, ==, 1);
    munit_assert_uint64(exported[1].norders, ==, 2);
    munit_assert_uint64(exported[2].norders, ==, 3);
    munit_assert_uint64(exported[3].norders, ==, 4);
    return MUNIT_OK;
}

/** @test Ask seek start finds the insertion window with a wrapped ascending ladder. */
static MunitResult test_ask_seek_start_wrapped_head(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[12];
    OrderbookLadderData ladder = {
        .num_levels = 10,
        .max_levels = 12,
        .levels = levels,
        .head = 9,
        .is_price_ascending = 1
    };

    for (uint64_t i = 0; i < ladder.num_levels; i++) {
        set_entry(&ladder, i, 100.0 + ((double)i * 0.01));
    }

    munit_assert_uint64(c_ladder_ask_lower_bound(&ladder, PRICE_TO_TICKS(100.05), 0, ladder.num_levels), ==, 5);
    munit_assert_uint64(c_ladder_ask_seek_start(&ladder, PRICE_TO_TICKS(100.09)), ==, 9);
    return MUNIT_OK;
}

/** @test Bid seek start finds the insertion window with a wrapped descending ladder. */
static MunitResult test_bid_seek_start_wrapped_head(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[12];
    OrderbookLadderData ladder = {
        .num_levels = 10,
        .max_levels = 12,
        .levels = levels,
        .head = 9,
        .is_price_ascending = 0
    };

    for (uint64_t i = 0; i < ladder.num_levels; i++) {
        set_entry(&ladder, i, 100.09 - ((double)i * 0.01));
    }

    munit_assert_uint64(c_ladder_bid_lower_bound(&ladder, PRICE_TO_TICKS(100.04), 0, ladder.num_levels), ==, 5);
    munit_assert_uint64(c_ladder_bid_seek_start(&ladder, PRICE_TO_TICKS(100.00)), ==, 9);
    return MUNIT_OK;
}

/** @test Sorted ask deltas update, delete, and insert while preserving ascending order. */
static MunitResult test_apply_sorted_deltas_ascending(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookEntry scratch[5];
    OrderbookEntry updates[3] = {
        make_entry(100.5, 1.5, 2),
        make_entry(101.0, 0.0, 1),
        make_entry(103.0, 2.0, 1),
    };
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 100.0);
    set_entry(&ladder, 1, 101.0);
    set_entry(&ladder, 2, 102.0);

    c_ladder_apply_sorted_deltas(&ladder, updates, 3, scratch);

    munit_assert_uint64(ladder.num_levels, ==, 4);
    munit_assert_uint64(ladder.head, ==, 0);
    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.0));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.5));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(102.0));
    munit_assert_uint64(c_ladder_at(&ladder, 3)->ticks, ==, PRICE_TO_TICKS(103.0));
    return MUNIT_OK;
}

/** @test Sorted bid deltas update, delete, and insert while preserving descending order. */
static MunitResult test_apply_sorted_deltas_descending(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[5];
    OrderbookEntry scratch[5];
    OrderbookEntry updates[3] = {
        make_entry(100.04, 1.5, 2),
        make_entry(100.02, 0.0, 1),
        make_entry(100.00, 2.0, 1),
    };
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 0
    };

    set_entry(&ladder, 0, 100.03);
    set_entry(&ladder, 1, 100.02);
    set_entry(&ladder, 2, 100.01);

    c_ladder_apply_sorted_deltas(&ladder, updates, 3, scratch);

    munit_assert_uint64(ladder.num_levels, ==, 4);
    munit_assert_uint64(ladder.head, ==, 0);
    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.04));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.03));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(100.01));
    munit_assert_uint64(c_ladder_at(&ladder, 3)->ticks, ==, PRICE_TO_TICKS(100.00));
    return MUNIT_OK;
}

/** @test Sorted bid delta merge truncates the worst tail when at capacity. */
static MunitResult test_apply_sorted_deltas_descending_capacity_truncates_tail(
    const MunitParameter params[] MUNIT_UNUSED,
    void* data MUNIT_UNUSED
) {
    OrderbookEntry levels[3];
    OrderbookEntry scratch[3];
    OrderbookEntry updates[1] = {
        make_entry(100.04, 1.5, 2),
    };
    OrderbookLadderData ladder = {
        .num_levels = 3,
        .max_levels = 3,
        .levels = levels,
        .head = 0,
        .is_price_ascending = 0
    };

    set_entry(&ladder, 0, 100.03);
    set_entry(&ladder, 1, 100.02);
    set_entry(&ladder, 2, 100.01);

    c_ladder_apply_sorted_deltas(&ladder, updates, 1, scratch);

    munit_assert_uint64(ladder.num_levels, ==, 3);
    munit_assert_uint64(ladder.head, ==, 0);
    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(100.04));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(100.03));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(100.02));
    return MUNIT_OK;
}

/** @test Sorted deltas preserve order and head position when the ladder is wrapped. */
static MunitResult test_apply_sorted_deltas_wrapped_head(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry levels[6];
    OrderbookEntry scratch[6];
    OrderbookEntry updates[3] = {
        make_entry(99.50, 1.5, 2),
        make_entry(100.01, 0.0, 1),
        make_entry(100.04, 2.0, 3),
    };
    OrderbookLadderData ladder = {
        .num_levels = 4,
        .max_levels = 6,
        .levels = levels,
        .head = 4,
        .is_price_ascending = 1
    };

    set_entry(&ladder, 0, 99.00);
    set_entry(&ladder, 1, 100.00);
    set_entry(&ladder, 2, 100.01);
    set_entry(&ladder, 3, 100.03);

    c_ladder_apply_sorted_deltas(&ladder, updates, 3, scratch);

    munit_assert_uint64(ladder.num_levels, ==, 5);
    munit_assert_uint64(ladder.head, ==, 0);
    munit_assert_uint64(c_ladder_at(&ladder, 0)->ticks, ==, PRICE_TO_TICKS(99.00));
    munit_assert_uint64(c_ladder_at(&ladder, 1)->ticks, ==, PRICE_TO_TICKS(99.50));
    munit_assert_uint64(c_ladder_at(&ladder, 2)->ticks, ==, PRICE_TO_TICKS(100.00));
    munit_assert_uint64(c_ladder_at(&ladder, 3)->ticks, ==, PRICE_TO_TICKS(100.03));
    munit_assert_uint64(c_ladder_at(&ladder, 4)->ticks, ==, PRICE_TO_TICKS(100.04));
    munit_assert_uint64(c_ladder_at(&ladder, 4)->norders, ==, 3);
    return MUNIT_OK;
}

static MunitTest ladder_tests[] = {
    { "/roll_right/at_start", test_roll_right_at_start, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/in_middle", test_roll_right_in_middle, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/uses_prefix_shift", test_roll_right_uses_prefix_shift, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/at_max_capacity", test_roll_right_at_max_capacity, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/beyond_count", test_roll_right_beyond_count, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/at_start", test_roll_left_at_start, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/in_middle", test_roll_left_in_middle, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/uses_prefix_shift", test_roll_left_uses_prefix_shift, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/at_end", test_roll_left_at_end, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/beyond_count", test_roll_left_beyond_count, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/insert_entry/basic", test_insert_entry, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/top_and_bottom/wrapped_head", test_top_and_bottom_wrapped_head, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/export_levels/wrapped_head", test_export_levels_wrapped_head, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/seek_start/ask_wrapped_head", test_ask_seek_start_wrapped_head, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/seek_start/bid_wrapped_head", test_bid_seek_start_wrapped_head, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/apply_sorted_deltas/ascending", test_apply_sorted_deltas_ascending, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/apply_sorted_deltas/descending", test_apply_sorted_deltas_descending, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/apply_sorted_deltas/descending_capacity_truncates_tail", test_apply_sorted_deltas_descending_capacity_truncates_tail, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/apply_sorted_deltas/wrapped_head", test_apply_sorted_deltas_wrapped_head, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/orderbook_entry_assign/pad_zero", test_orderbook_entry_assign_pad_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

const MunitSuite ladder_suite = {
    "/ladder",
    ladder_tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};

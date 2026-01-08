/**
 * test_ladder.c - µnit tests for orderbook_ladder.c
 *
 * Tests ladder operations: roll_right, roll_left, and insert_level.
 */

#include "munit.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_ladder.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_types.h"
#include <stdlib.h>
#include <string.h>

#define TICK_SIZE 0.01
#define LOT_SIZE 0.001

/* ============================================================================
 * Helper functions
 * ============================================================================ */

static OrderbookLevel make_level(double price, double size, uint64_t norders) {
    OrderbookLevel level = {
        .price = price,
        .size = size,
        .norders = norders,
        .ticks = (uint64_t)(price / TICK_SIZE),
        .lots = (uint64_t)(size / LOT_SIZE)
    };
    return level;
}

/* ============================================================================
 * roll_right tests
 * ============================================================================ */

static MunitResult test_roll_right_at_start(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 2,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };

    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);

    c_ladder_roll_right(&ladder_data, 0);
    
    // Should have shifted: [100, 101] -> [?, 100, 101]
    // Insert new level at 0
    OrderbookLevel new_level = make_level(99.0, 1.0, 1);
    c_ladder_insert_level(levels, 0, &new_level);
    ladder_data.num_levels = 3;

    munit_assert_double_equal(levels[0].price, 99.0, 10);
    munit_assert_double_equal(levels[1].price, 100.0, 10);
    munit_assert_double_equal(levels[2].price, 101.0, 10);
    return MUNIT_OK;
}

static MunitResult test_roll_right_in_middle(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);
    levels[2] = make_level(102.0, 1.0, 1);
    
    c_ladder_roll_right(&ladder_data, 1);

    // Should have shifted from index 1: [100, 101, 102] -> [100, ?, 101, 102]
    OrderbookLevel new_level = make_level(100.5, 2.0, 1);
    c_ladder_insert_level(levels, 1, &new_level);
    ladder_data.num_levels = 4;

    munit_assert_double_equal(levels[0].price, 100.0, 10);
    munit_assert_double_equal(levels[1].size, 2.0, 10);  // New level
    munit_assert_double_equal(levels[2].price, 101.0, 10);
    return MUNIT_OK;
}

static MunitResult test_roll_right_at_max_capacity(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[3];
    OrderbookLadderData ladder_data = {
        .num_levels = 3,
        .max_levels = 3,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(100.01, 1.0, 1);
    levels[2] = make_level(100.02, 1.0, 1);
    
    c_ladder_roll_right(&ladder_data, 0);
    
    // At max capacity, last element should be dropped
    OrderbookLevel new_level = make_level(99.0, 2.0, 1);
    c_ladder_insert_level(levels, 0, &new_level);
    // Don't increment num_levels - we're replacing dropped element
    
    munit_assert_double_equal(levels[0].price, 99.0, 10);
    munit_assert_double_equal(levels[1].price, 100.0, 10);
    // levels[2] should still be 100.02 (or 100.01 if shifted)
    return MUNIT_OK;
}

static MunitResult test_roll_right_beyond_count(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 2,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);
    
    // Should be no-op if start_index > num_levels
    c_ladder_roll_right(&ladder_data, 5);
    
    munit_assert_double_equal(levels[0].price, 100.0, 10);
    munit_assert_double_equal(levels[1].price, 101.0, 10);
    return MUNIT_OK;
}

/* ============================================================================
 * roll_left tests
 * ============================================================================ */

static MunitResult test_roll_left_at_start(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);
    levels[2] = make_level(102.0, 1.0, 1);
    
    c_ladder_roll_left(&ladder_data, 0);
    ladder_data.num_levels = 2;
    
    // Should have removed first element: [100, 101, 102] -> [101, 102]
    munit_assert_double_equal(levels[0].price, 101.0, 10);
    munit_assert_double_equal(levels[1].price, 102.0, 10);
    return MUNIT_OK;
}

static MunitResult test_roll_left_in_middle(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);
    levels[2] = make_level(102.0, 1.0, 1);
    
    c_ladder_roll_left(&ladder_data, 1);
    ladder_data.num_levels = 2;
    
    // Should have removed middle element: [100, 101, 102] -> [100, 102]
    munit_assert_double_equal(levels[0].price, 100.0, 10);
    munit_assert_double_equal(levels[1].price, 102.0, 10);
    return MUNIT_OK;
}

static MunitResult test_roll_left_at_end(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 3,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);
    levels[2] = make_level(102.0, 1.0, 1);
    
    c_ladder_roll_left(&ladder_data, 2);
    ladder_data.num_levels = 2;
    
    // Should have removed last element: [100, 101, 102] -> [100, 101]
    munit_assert_double_equal(levels[0].price, 100.0, 10);
    munit_assert_double_equal(levels[1].price, 101.0, 10);
    return MUNIT_OK;
}

static MunitResult test_roll_left_beyond_count(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLadderData ladder_data = {
        .num_levels = 2,
        .max_levels = 5,
        .levels = levels,
        .is_price_ascending = 1
    };
    
    levels[0] = make_level(100.0, 1.0, 1);
    levels[1] = make_level(101.0, 1.0, 1);
    
    // Should be no-op if start_index >= num_levels
    c_ladder_roll_left(&ladder_data, 5);
    
    munit_assert_double_equal(levels[0].price, 100.0, 10);
    munit_assert_double_equal(levels[1].price, 101.0, 10);
    return MUNIT_OK;
}

/* ============================================================================
 * insert_level tests
 * ============================================================================ */

static MunitResult test_insert_level(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    OrderbookLevel new_level = make_level(100.5, 2.0, 3);
    
    c_ladder_insert_level(levels, 0, &new_level);
    
    munit_assert_double_equal(levels[0].price, 100.5, 10);
    munit_assert_double_equal(levels[0].size, 2.0, 10);
    munit_assert_uint64(levels[0].norders, ==, 3);
    return MUNIT_OK;
}

static MunitResult test_insert_level_multiple(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[5];
    
    for (uint64_t i = 0; i < 3; i++) {
        OrderbookLevel level = make_level(100.0 + i * 0.01, (double)(i + 1), 1);
        c_ladder_insert_level(levels, i, &level);
    }
    
    munit_assert_double_equal(levels[0].price, 100.0, 10);
    munit_assert_double_equal(levels[1].price, 100.01, 10);
    munit_assert_double_equal(levels[2].price, 100.02, 10);
    return MUNIT_OK;
}

/* ============================================================================
 * Test suite definition
 * ============================================================================ */

static MunitTest ladder_tests[] = {
    { "/roll_right/at_start", test_roll_right_at_start, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/in_middle", test_roll_right_in_middle, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/at_max_capacity", test_roll_right_at_max_capacity, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_right/beyond_count", test_roll_right_beyond_count, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/at_start", test_roll_left_at_start, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/in_middle", test_roll_left_in_middle, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/at_end", test_roll_left_at_end, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/roll_left/beyond_count", test_roll_left_beyond_count, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/insert_level/basic", test_insert_level, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/insert_level/multiple", test_insert_level_multiple, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

const MunitSuite ladder_suite = {
    "/ladder",
    ladder_tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};


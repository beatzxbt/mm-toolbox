/**
 * test_helpers.c - µnit tests for orderbook_helpers.c
 *
 * Tests conversion functions (price<->tick, size<->lot) and
 * level manipulation functions (swap, reverse, sort).
 */

#include "munit.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_helpers.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TICK_SIZE 0.01
#define LOT_SIZE 0.001

static int approx_eq(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

/* ============================================================================
 * Conversion function tests
 * ============================================================================ */

static MunitResult test_price_to_tick_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t ticks = price_to_tick(100.01, TICK_SIZE);
    munit_assert_uint64(ticks, ==, 10001);
    return MUNIT_OK;
}

static MunitResult test_price_to_tick_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t ticks = price_to_tick(0.0, TICK_SIZE);
    munit_assert_uint64(ticks, ==, 0);
    return MUNIT_OK;
}

static MunitResult test_price_to_tick_rounding(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    // 100.005 / 0.01 = 10000.5, should floor to 10000
    uint64_t ticks = price_to_tick(100.005, TICK_SIZE);
    munit_assert_uint64(ticks, ==, 10000);
    return MUNIT_OK;
}

static MunitResult test_tick_to_price_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double price = tick_to_price(10001, TICK_SIZE);
    munit_assert_double_equal(price, 100.01, 10);
    return MUNIT_OK;
}

static MunitResult test_tick_to_price_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double price = tick_to_price(0, TICK_SIZE);
    munit_assert_double_equal(price, 0.0, 10);
    return MUNIT_OK;
}

static MunitResult test_tick_conversion_roundtrip(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double original = 123.45;
    uint64_t ticks = price_to_tick(original, TICK_SIZE);
    double recovered = tick_to_price(ticks, TICK_SIZE);
    munit_assert_true(approx_eq(original, recovered, 1e-9));
    return MUNIT_OK;
}

static MunitResult test_size_to_lot_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t lots = size_to_lot(1.5, LOT_SIZE);
    munit_assert_uint64(lots, ==, 1500);
    return MUNIT_OK;
}

static MunitResult test_size_to_lot_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t lots = size_to_lot(0.0, LOT_SIZE);
    munit_assert_uint64(lots, ==, 0);
    return MUNIT_OK;
}

static MunitResult test_lot_to_size_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double size = lot_to_size(1500, LOT_SIZE);
    munit_assert_double_equal(size, 1.5, 10);
    return MUNIT_OK;
}

static MunitResult test_lot_conversion_roundtrip(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double original = 99.999;
    uint64_t lots = size_to_lot(original, LOT_SIZE);
    double recovered = lot_to_size(lots, LOT_SIZE);
    munit_assert_true(approx_eq(original, recovered, 1e-9));
    return MUNIT_OK;
}

/* ============================================================================
 * Level manipulation function tests
 * ============================================================================ */

static MunitResult test_swap_levels(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel a = {.price = 100.0, .size = 1.0, .norders = 1, .ticks = 10000, .lots = 1000};
    OrderbookLevel b = {.price = 200.0, .size = 2.0, .norders = 2, .ticks = 20000, .lots = 2000};
    
    swap_levels(&a, &b);
    
    munit_assert_double_equal(a.price, 200.0, 10);
    munit_assert_double_equal(a.size, 2.0, 10);
    munit_assert_uint64(a.norders, ==, 2);
    munit_assert_double_equal(b.price, 100.0, 10);
    munit_assert_double_equal(b.size, 1.0, 10);
    munit_assert_uint64(b.norders, ==, 1);
    return MUNIT_OK;
}

static MunitResult test_reverse_levels_empty(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[1];
    reverse_levels_inplace(0, levels);
    // Should not crash
    return MUNIT_OK;
}

static MunitResult test_reverse_levels_single(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[1] = {{.price = 100.0, .size = 1.0, .norders = 1, .ticks = 10000, .lots = 1000}};
    reverse_levels_inplace(1, levels);
    munit_assert_double_equal(levels[0].price, 100.0, 10);
    return MUNIT_OK;
}

static MunitResult test_reverse_levels_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[3] = {
        {.price = 100.0, .size = 1.0, .norders = 1, .ticks = 10000, .lots = 1000},
        {.price = 101.0, .size = 2.0, .norders = 2, .ticks = 10100, .lots = 2000},
        {.price = 102.0, .size = 3.0, .norders = 3, .ticks = 10200, .lots = 3000}
    };
    
    reverse_levels_inplace(3, levels);
    
    munit_assert_double_equal(levels[0].price, 102.0, 10);
    munit_assert_double_equal(levels[1].price, 101.0, 10);
    munit_assert_double_equal(levels[2].price, 100.0, 10);
    return MUNIT_OK;
}

static MunitResult test_reverse_levels_even(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[4] = {
        {.price = 1.0, .ticks = 100, .size = 1.0, .lots = 1000},
        {.price = 2.0, .ticks = 200, .size = 1.0, .lots = 1000},
        {.price = 3.0, .ticks = 300, .size = 1.0, .lots = 1000},
        {.price = 4.0, .ticks = 400, .size = 1.0, .lots = 1000}
    };
    
    reverse_levels_inplace(4, levels);
    
    munit_assert_double_equal(levels[0].price, 4.0, 10);
    munit_assert_double_equal(levels[1].price, 3.0, 10);
    munit_assert_double_equal(levels[2].price, 2.0, 10);
    munit_assert_double_equal(levels[3].price, 1.0, 10);
    return MUNIT_OK;
}

static MunitResult test_is_sorted_ascending_true(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[3] = {
        {.ticks = 10000},
        {.ticks = 10100},
        {.ticks = 10200}
    };
    munit_assert_true(is_sorted_by_tick(3, levels, true));
    return MUNIT_OK;
}

static MunitResult test_is_sorted_ascending_false(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[3] = {
        {.ticks = 10300},
        {.ticks = 10100},
        {.ticks = 10200}
    };
    munit_assert_false(is_sorted_by_tick(3, levels, true));
    return MUNIT_OK;
}

static MunitResult test_is_sorted_descending_true(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[3] = {
        {.ticks = 10300},
        {.ticks = 10200},
        {.ticks = 10100}
    };
    munit_assert_true(is_sorted_by_tick(3, levels, false));
    return MUNIT_OK;
}

static MunitResult test_is_sorted_empty(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[1];
    munit_assert_true(is_sorted_by_tick(0, levels, true));
    munit_assert_true(is_sorted_by_tick(1, levels, true));
    return MUNIT_OK;
}

static MunitResult test_sort_levels_ascending(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[4] = {
        {.price = 103.0, .ticks = 10300},
        {.price = 101.0, .ticks = 10100},
        {.price = 104.0, .ticks = 10400},
        {.price = 102.0, .ticks = 10200}
    };
    
    sort_levels_by_tick(4, levels, true);
    
    munit_assert_uint64(levels[0].ticks, ==, 10100);
    munit_assert_uint64(levels[1].ticks, ==, 10200);
    munit_assert_uint64(levels[2].ticks, ==, 10300);
    munit_assert_uint64(levels[3].ticks, ==, 10400);
    return MUNIT_OK;
}

static MunitResult test_sort_levels_descending(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[4] = {
        {.price = 101.0, .ticks = 10100},
        {.price = 103.0, .ticks = 10300},
        {.price = 100.0, .ticks = 10000},
        {.price = 102.0, .ticks = 10200}
    };
    
    sort_levels_by_tick(4, levels, false);
    
    munit_assert_uint64(levels[0].ticks, ==, 10300);
    munit_assert_uint64(levels[1].ticks, ==, 10200);
    munit_assert_uint64(levels[2].ticks, ==, 10100);
    munit_assert_uint64(levels[3].ticks, ==, 10000);
    return MUNIT_OK;
}

static MunitResult test_sort_levels_already_sorted(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookLevel levels[3] = {
        {.price = 100.0, .ticks = 10000},
        {.price = 101.0, .ticks = 10100},
        {.price = 102.0, .ticks = 10200}
    };
    
    sort_levels_by_tick(3, levels, true);
    
    // Should remain sorted
    munit_assert_uint64(levels[0].ticks, ==, 10000);
    munit_assert_uint64(levels[1].ticks, ==, 10100);
    munit_assert_uint64(levels[2].ticks, ==, 10200);
    return MUNIT_OK;
}

/* ============================================================================
 * Test suite definition
 * ============================================================================ */

static MunitTest helper_tests[] = {
    { "/price_to_tick/basic", test_price_to_tick_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/price_to_tick/zero", test_price_to_tick_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/price_to_tick/rounding", test_price_to_tick_rounding, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/tick_to_price/basic", test_tick_to_price_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/tick_to_price/zero", test_tick_to_price_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/tick_conversion/roundtrip", test_tick_conversion_roundtrip, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/size_to_lot/basic", test_size_to_lot_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/size_to_lot/zero", test_size_to_lot_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/lot_to_size/basic", test_lot_to_size_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/lot_conversion/roundtrip", test_lot_conversion_roundtrip, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/swap_levels", test_swap_levels, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/reverse_levels/empty", test_reverse_levels_empty, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/reverse_levels/single", test_reverse_levels_single, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/reverse_levels/basic", test_reverse_levels_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/reverse_levels/even", test_reverse_levels_even, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/is_sorted/ascending/true", test_is_sorted_ascending_true, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/is_sorted/ascending/false", test_is_sorted_ascending_false, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/is_sorted/descending/true", test_is_sorted_descending_true, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/is_sorted/empty", test_is_sorted_empty, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/sort_levels/ascending", test_sort_levels_ascending, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/sort_levels/descending", test_sort_levels_descending, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/sort_levels/already_sorted", test_sort_levels_already_sorted, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

const MunitSuite helpers_suite = {
    "/helpers",
    helper_tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};


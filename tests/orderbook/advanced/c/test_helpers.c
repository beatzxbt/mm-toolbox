/**
 * test_helpers.c - µnit tests for orderbook_helpers.c
 *
 * Tests conversion functions (price<->tick, size<->lot).
 */

#include "munit.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_helpers.h"
#include "../../../../src/mm_toolbox/orderbook/advanced/c/orderbook_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TICK_SIZE 0.01
#define LOT_SIZE 0.001
#define TICK_SIZE_RECIP (1.0 / TICK_SIZE)
#define LOT_SIZE_RECIP (1.0 / LOT_SIZE)

/**
 * @brief Check if two doubles are approximately equal.
 *
 * @param a First value.
 * @param b Second value.
 * @param tol Absolute tolerance.
 * @return 1 if |a - b| < tol, else 0.
 */
static int approx_eq(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

/* ============================================================================
 * Conversion function tests
 * ============================================================================ */

/**
 * @test Price to tick conversion: 100.01 with reciprocal tick size yields 10001 ticks.
 */
static MunitResult test_price_to_tick_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t ticks = price_to_tick(100.01, TICK_SIZE_RECIP);
    munit_assert_uint64(ticks, ==, 10001);
    return MUNIT_OK;
}

/**
 * @test Price to tick conversion with zero price yields 0 ticks.
 */
static MunitResult test_price_to_tick_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t ticks = price_to_tick(0.0, TICK_SIZE_RECIP);
    munit_assert_uint64(ticks, ==, 0);
    return MUNIT_OK;
}

/**
 * @test Price to tick rounding: 100.005 / 0.01 = 10000.5 floors to 10000.
 */
static MunitResult test_price_to_tick_rounding(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    // 100.005 / 0.01 = 10000.5, should floor to 10000
    uint64_t ticks = price_to_tick(100.005, TICK_SIZE_RECIP);
    munit_assert_uint64(ticks, ==, 10000);
    return MUNIT_OK;
}

/** @test Exact large integers stay exact; only near-integer noise is corrected. */
static MunitResult test_scaled_conversion_boundaries(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    munit_assert_uint64(price_to_tick(1e19, 1.0), ==, UINT64_C(10000000000000000000));
    munit_assert_uint64(size_to_lot(1.2e19, 1.0), ==, UINT64_C(12000000000000000000));
    munit_assert_uint64(price_to_tick(nextafter(9.0, 0.0), 1.0), ==, 9);
    munit_assert_uint64(size_to_lot(8.5, 1.0), ==, 8);
    return MUNIT_OK;
}

/** @test The noexcept C conversion sentinel prevents out-of-range integer casts. */
static MunitResult test_scaled_conversion_invalid_sentinel(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    const double invalid[] = {-1.0, NAN, INFINITY, 18446744073709551616.0};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        munit_assert_uint64(price_to_tick(invalid[i], 1.0), ==, 0);
        munit_assert_uint64(size_to_lot(invalid[i], 1.0), ==, 0);
    }
    return MUNIT_OK;
}

/**
 * @test Tick to price conversion: 10001 ticks with 0.01 tick size yields 100.01.
 */
static MunitResult test_tick_to_price_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double price = tick_to_price(10001, TICK_SIZE);
    munit_assert_double_equal(price, 100.01, 10);
    return MUNIT_OK;
}

/**
 * @test Tick to price conversion with zero ticks yields 0.0.
 */
static MunitResult test_tick_to_price_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double price = tick_to_price(0, TICK_SIZE);
    munit_assert_double_equal(price, 0.0, 10);
    return MUNIT_OK;
}

/**
 * @test Roundtrip: price -> tick -> price recovers original value.
 */
static MunitResult test_tick_conversion_roundtrip(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double original = 123.45;
    uint64_t ticks = price_to_tick(original, TICK_SIZE_RECIP);
    double recovered = tick_to_price(ticks, TICK_SIZE);
    munit_assert_true(approx_eq(original, recovered, 1e-9));
    return MUNIT_OK;
}

/**
 * @test Size to lot conversion: 1.5 with 0.001 lot size yields 1500 lots.
 */
static MunitResult test_size_to_lot_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t lots = size_to_lot(1.5, LOT_SIZE_RECIP);
    munit_assert_uint64(lots, ==, 1500);
    return MUNIT_OK;
}

/**
 * @test Size to lot conversion with zero size yields 0 lots.
 */
static MunitResult test_size_to_lot_zero(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    uint64_t lots = size_to_lot(0.0, LOT_SIZE_RECIP);
    munit_assert_uint64(lots, ==, 0);
    return MUNIT_OK;
}

/**
 * @test Lot to size conversion: 1500 lots with 0.001 lot size yields 1.5.
 */
static MunitResult test_lot_to_size_basic(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double size = lot_to_size(1500, LOT_SIZE);
    munit_assert_double_equal(size, 1.5, 10);
    return MUNIT_OK;
}

/**
 * @test Roundtrip: size -> lot -> size recovers original value.
 */
static MunitResult test_lot_conversion_roundtrip(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    double original = 99.999;
    uint64_t lots = size_to_lot(original, LOT_SIZE_RECIP);
    double recovered = lot_to_size(lots, LOT_SIZE);
    munit_assert_true(approx_eq(original, recovered, 1e-9));
    return MUNIT_OK;
}

/** @test Snapshot compaction retains final original quantities before reversal. */
static MunitResult test_compact_entries_snapshot(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    OrderbookEntry entries[] = {
        {.ticks = 120, .lots = 5, .norders = 5},
        {.ticks = 110, .lots = 4, .norders = 4},
        {.ticks = 110, .lots = 0, .norders = 0},
        {.ticks = 100, .lots = 2, .norders = 2},
        {.ticks = 100, .lots = 7, .norders = 7},
    };
    uint64_t count = compact_entries(5, entries, true);
    munit_assert_uint64(count, ==, 2);
    reverse_entries(count, entries);
    munit_assert_uint64(entries[0].ticks, ==, 100);
    munit_assert_uint64(entries[0].lots, ==, 7);
    munit_assert_uint64(entries[0].norders, ==, 7);
    munit_assert_uint64(entries[1].ticks, ==, 120);
    munit_assert_uint64(entries[1].lots, ==, 5);
    munit_assert_uint64(compact_entries(0, NULL, true), ==, 0);
    return MUNIT_OK;
}

/** @test Fractional weighted ticks floor, including highly skewed ordinary prices. */
static MunitResult test_weighted_mid_floor(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 1, 1), ==, 102);
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 1, 2), ==, 103);
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 18, 2), ==, 100);
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 1, UINT64_C(10000000000000000)), ==, 104);
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, UINT64_C(10000000000000000), 1), ==, 100);
    return MUNIT_OK;
}

/** @test Exact large-price skew mirrors cannot round beyond either touch. */
static MunitResult test_weighted_mid_large_skew(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    const uint64_t dominant = UINT64_C(10000000000000000);
    munit_assert_uint64(orderbook_weighted_mid_tick(
        UINT64_C(10000000000000000000), UINT64_C(12000000000000000000), 1, dominant
    ), ==, UINT64_C(11999999999999999800));
    munit_assert_uint64(orderbook_weighted_mid_tick(
        UINT64_C(10000000000000000000), UINT64_C(12000000000000000000), dominant, 1
    ), ==, UINT64_C(10000000000000000199));
    munit_assert_uint64(orderbook_weighted_mid_tick(
        UINT64_C(16000000000000000000), UINT64_C(18000000000000000000), 1, dominant
    ), ==, UINT64_C(17999999999999999800));
    munit_assert_uint64(orderbook_weighted_mid_tick(
        UINT64_C(16000000000000000000), UINT64_C(18000000000000000000), dominant, 1
    ), ==, UINT64_C(16000000000000000199));
    return MUNIT_OK;
}

/** @test Near the integer limit, no floating conversion can overflow or escape the BBO. */
static MunitResult test_weighted_mid_near_u64(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    munit_assert_uint64(orderbook_weighted_mid_tick(
        UINT64_MAX - 1000, UINT64_MAX, 1, UINT64_MAX
    ), ==, UINT64_MAX - 1);
    munit_assert_uint64(orderbook_weighted_mid_tick(
        UINT64_MAX - 1000, UINT64_MAX, UINT64_MAX, 1
    ), ==, UINT64_MAX - 1000);
    return MUNIT_OK;
}

/** @test A 65-bit size denominator and 128-bit product retain exact floor semantics. */
static MunitResult test_weighted_mid_large_denominator(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 111, UINT64_MAX, UINT64_MAX), ==, 105);
    munit_assert_uint64(orderbook_weighted_mid_tick(0, UINT64_MAX, UINT64_MAX, UINT64_MAX), ==, UINT64_MAX / 2);
    munit_assert_uint64(orderbook_weighted_mid_tick(0, UINT64_MAX, UINT64_MAX, 1), ==, 0);
    munit_assert_uint64(orderbook_weighted_mid_tick(0, UINT64_MAX, 1, UINT64_MAX), ==, UINT64_MAX - 1);
    return MUNIT_OK;
}

/** @test Zero quantities return the supported endpoint, or zero for no size. */
static MunitResult test_weighted_mid_zero_sizes(const MunitParameter params[] MUNIT_UNUSED, void* data MUNIT_UNUSED) {
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 0, 0), ==, 0);
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 0, 1), ==, 105);
    munit_assert_uint64(orderbook_weighted_mid_tick(100, 105, 1, 0), ==, 100);
    munit_assert_uint64(orderbook_weighted_mid_tick(UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX), ==, UINT64_MAX);
    return MUNIT_OK;
}

/* ============================================================================
 * Test suite definition
 * ============================================================================ */

static MunitTest helper_tests[] = {
    { "/weighted_mid/floor", test_weighted_mid_floor, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/weighted_mid/large_skew", test_weighted_mid_large_skew, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/weighted_mid/near_u64", test_weighted_mid_near_u64, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/weighted_mid/large_denominator", test_weighted_mid_large_denominator, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/weighted_mid/zero_sizes", test_weighted_mid_zero_sizes, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/compact_entries/snapshot_last_wins", test_compact_entries_snapshot, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/price_to_tick/basic", test_price_to_tick_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/price_to_tick/zero", test_price_to_tick_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/price_to_tick/rounding", test_price_to_tick_rounding, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/conversion/scaled_boundaries", test_scaled_conversion_boundaries, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/conversion/invalid_sentinel", test_scaled_conversion_invalid_sentinel, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/tick_to_price/basic", test_tick_to_price_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/tick_to_price/zero", test_tick_to_price_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/tick_conversion/roundtrip", test_tick_conversion_roundtrip, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/size_to_lot/basic", test_size_to_lot_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/size_to_lot/zero", test_size_to_lot_zero, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/lot_to_size/basic", test_lot_to_size_basic, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { "/lot_conversion/roundtrip", test_lot_conversion_roundtrip, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL },
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

const MunitSuite helpers_suite = {
    "/helpers",
    helper_tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};

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
    { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

const MunitSuite helpers_suite = {
    "/helpers",
    helper_tests,
    NULL,
    1,
    MUNIT_SUITE_OPTION_NONE
};

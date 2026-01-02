/**
 * test_runner.c - Main test runner combining all C test suites
 */

#include "munit.h"

/* Import test suites */
extern const MunitSuite helpers_suite;
extern const MunitSuite ladder_suite;

/* Combine all suites */
static MunitSuite suites[3];

static const MunitSuite root_suite = {
    "/orderbook",
    NULL,
    suites,
    1,
    MUNIT_SUITE_OPTION_NONE
};

int main(int argc, char* argv[]) {
    suites[0] = helpers_suite;
    suites[1] = ladder_suite;
    suites[2] = (MunitSuite){ NULL, NULL, NULL, 0, MUNIT_SUITE_OPTION_NONE };
    return munit_suite_main(&root_suite, NULL, argc, argv);
}

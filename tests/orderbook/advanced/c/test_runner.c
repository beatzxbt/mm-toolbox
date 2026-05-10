/**
 * @file test_runner.c
 * @brief Main test runner combining all C test suites for the orderbook module.
 *
 * Aggregates helpers_suite and ladder_suite under a single root suite
 * and delegates to the µnit framework for execution.
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

/**
 * @brief Entry point for the orderbook C test suite.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Exit code from munit_suite_main.
 */
int main(int argc, char* argv[]) {
    suites[0] = helpers_suite;
    suites[1] = ladder_suite;
    suites[2] = (MunitSuite){ NULL, NULL, NULL, 0, MUNIT_SUITE_OPTION_NONE };
    return munit_suite_main(&root_suite, NULL, argc, argv);
}

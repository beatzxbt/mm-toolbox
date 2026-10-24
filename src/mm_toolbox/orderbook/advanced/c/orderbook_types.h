/**
 * @file orderbook_types.h
 * @brief C struct definitions for orderbook components.
 *
 * These structs match the Cython definitions in level/level.pxd exactly,
 * enabling seamless interoperability between C and Cython code.
 */

#ifndef ORDERBOOK_TYPES_H
#define ORDERBOOK_TYPES_H

#include <stdint.h>

/**
 * @brief Maximum number of orderbook levels to prevent integer overflow.
 */
#define ORDERBOOK_MAX_LEVELS (16777216UL)

/**
 * @brief Public/raw price level used at API boundaries.
 *
 * Fields:
 *   price   - Raw floating-point price.
 *   size    - Raw floating-point size.
 *   norders - Number of orders at this level.
 */
typedef struct {
    double price;
    double size;
    uint64_t norders;
} OrderbookLevel;

/**
 * @brief Compact normalized level used internally by the core and ladders.
 *
 * Fields:
 *   ticks   - Price expressed in integer tick units.
 *   lots    - Size expressed in integer lot units.
 *   norders - Number of orders at this level.
 *   _pad    - Reserved padding for 32-byte alignment and deterministic tests.
 */
typedef struct {
    uint64_t ticks;
    uint64_t lots;
    uint64_t norders;
    uint64_t _pad;
} OrderbookEntry;

/**
 * @brief A collection of orderbook levels.
 *
 * Fields:
 *   num_levels - Number of valid levels in the array.
 *   levels     - Pointer to the array of OrderbookLevel structs.
 */
typedef struct {
    uint64_t num_levels;
    OrderbookLevel* levels;
} OrderbookLevels;

#endif /* ORDERBOOK_TYPES_H */

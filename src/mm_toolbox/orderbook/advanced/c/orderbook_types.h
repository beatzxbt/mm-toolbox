/**
 * orderbook_types.h - C struct definitions for orderbook components.
 *
 * These structs match the Cython definitions in level/level.pxd exactly,
 * enabling seamless interoperability between C and Cython code.
 */

#ifndef ORDERBOOK_TYPES_H
#define ORDERBOOK_TYPES_H

#include <stdint.h>

/**
 * Maximum number of orderbook levels to prevent integer overflow.
 *
 * Rationale: sizeof(OrderbookLevel) = 64 bytes, so:
 * - 1M levels = 64 MB (reasonable)
 * - 16M levels = 1 GB (max safe allocation)
 *
 * This limit prevents overflow in: num_levels * sizeof(OrderbookLevel)
 */
#define ORDERBOOK_MAX_LEVELS (16777216UL)  /* 2^24 = 16M levels */

/**
 * OrderbookLevel - A single price level in the orderbook.
 *
 * Fields:
 *   price     - The raw price as a floating-point value.
 *   size      - The total size/quantity at this level.
 *   norders   - Number of orders at this level.
 *   ticks     - Price converted to integer tick units.
 *   lots      - Size converted to integer lot units.
 *   __padding1-3 - Reserved for cache line alignment (64 bytes total).
 */
typedef struct {
    double price;
    double size;
    uint64_t norders;
    uint64_t ticks;
    uint64_t lots;
    uint64_t __padding1;
    uint64_t __padding2;
    uint64_t __padding3;
} OrderbookLevel;

/**
 * OrderbookLevels - A collection of orderbook levels.
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


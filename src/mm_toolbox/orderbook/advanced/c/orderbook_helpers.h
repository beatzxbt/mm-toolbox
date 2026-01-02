/**
 * orderbook_helpers.h - Helper functions for orderbook operations.
 *
 * Provides conversion utilities (price<->tick, size<->lot) and sorting
 * operations for OrderbookLevel arrays.
 */

#ifndef ORDERBOOK_HELPERS_H
#define ORDERBOOK_HELPERS_H

#include "orderbook_types.h"
#include <stdbool.h>

/* Conversion functions: price/size to integer tick/lot units */

/**
 * Convert a price to tick units using floor division.
 *
 * @param price     The price value to convert.
 * @param tick_size The size of one tick.
 * @return          The price expressed in ticks.
 */
uint64_t price_to_tick(double price, double tick_size);

/**
 * Convert a price to tick units using multiplication with pre-computed reciprocal.
 * Faster than price_to_tick() by avoiding division (2-3 cycles vs 10-20 cycles).
 *
 * @param price          The price value to convert.
 * @param tick_size_recip The reciprocal of tick_size (1.0 / tick_size).
 * @return               The price expressed in ticks.
 */
uint64_t price_to_tick_fast(double price, double tick_size_recip);

/**
 * Convert a size to lot units using floor division.
 *
 * @param size     The size value to convert.
 * @param lot_size The size of one lot.
 * @return         The size expressed in lots.
 */
uint64_t size_to_lot(double size, double lot_size);

/**
 * Convert a size to lot units using multiplication with pre-computed reciprocal.
 * Faster than size_to_lot() by avoiding division (2-3 cycles vs 10-20 cycles).
 *
 * @param size          The size value to convert.
 * @param lot_size_recip The reciprocal of lot_size (1.0 / lot_size).
 * @return              The size expressed in lots.
 */
uint64_t size_to_lot_fast(double size, double lot_size_recip);

/**
 * Convert tick units back to a price.
 *
 * @param tick      The tick value to convert.
 * @param tick_size The size of one tick.
 * @return          The price value.
 */
double tick_to_price(uint64_t tick, double tick_size);

/**
 * Convert lot units back to a size.
 *
 * @param lot      The lot value to convert.
 * @param lot_size The size of one lot.
 * @return         The size value.
 */
double lot_to_size(uint64_t lot, double lot_size);

/* Level manipulation functions */

/**
 * Swap two OrderbookLevel structs in place.
 *
 * @param a Pointer to the first level.
 * @param b Pointer to the second level.
 */
void swap_levels(OrderbookLevel* a, OrderbookLevel* b);

/**
 * Reverse the order of levels in an array in place.
 *
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 */
void reverse_levels_inplace(uint64_t num_levels, OrderbookLevel* levels);

/**
 * Check if levels are sorted by tick.
 *
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @param ascending  If true, check ascending order; if false, descending.
 * @return           True if sorted in the specified order, false otherwise.
 */
bool is_sorted_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending);

/**
 * Sort levels by tick in place using insertion sort.
 * Optimal for nearly-sorted data (O(n) best case, O(n²) worst case).
 *
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @param ascending  If true, sort ascending; if false, sort descending.
 */
void insertion_sort_levels_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending);

/**
 * Sort levels by tick in place with smart algorithm.
 * Checks if already sorted first, then uses insertion sort if needed.
 * Best for exchange data which is typically pre-sorted or nearly-sorted.
 *
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @param ascending  If true, sort ascending; if false, sort descending.
 */
void sort_levels_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending);

#endif /* ORDERBOOK_HELPERS_H */

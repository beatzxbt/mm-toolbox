/**
 * orderbook_ladder.h - Ladder operations for orderbook level arrays.
 *
 * Provides efficient memory shifting operations (roll_right, roll_left) and
 * direct level insertion for managing ordered price levels in the orderbook.
 */

#ifndef ORDERBOOK_LADDER_H
#define ORDERBOOK_LADDER_H

#include "orderbook_types.h"
#include <stdint.h>

/**
 * OrderbookLadderData - Internal data structure for a ladder's level array with metadata.
 *
 * Fields:
 *   num_levels         - Current number of valid levels in the array.
 *   max_levels         - Maximum capacity of the levels array.
 *   levels             - Pointer to the array of OrderbookLevel structs.
 *   is_price_ascending - Non-zero if prices are sorted ascending (asks).
 */
typedef struct {
    uint64_t num_levels;
    uint64_t max_levels;
    OrderbookLevel* levels;
    int is_price_ascending;
} OrderbookLadderData;

/**
 * Shift levels right starting from start_index to make room for insertion.
 *
 * When at max capacity, the last element is dropped. This only shifts data;
 * the caller must update num_levels separately.
 *
 * @param data        Pointer to the ladder data containing levels and metadata.
 * @param start_index Index from which to start shifting right.
 */
void c_ladder_roll_right(OrderbookLadderData* data, uint64_t start_index);

/**
 * Shift levels left starting from start_index to remove a level.
 *
 * This only shifts data; the caller must update num_levels separately.
 *
 * @param data        Pointer to the ladder data containing levels and metadata.
 * @param start_index Index from which to start shifting left.
 */
void c_ladder_roll_left(OrderbookLadderData* data, uint64_t start_index);

/**
 * Insert a level directly at the specified index.
 *
 * @param levels Pointer to the levels array.
 * @param index  Index at which to insert the level.
 * @param level  Pointer to the OrderbookLevel to insert.
 */
void c_ladder_insert_level(OrderbookLevel* levels, uint64_t index, const OrderbookLevel* level);

#endif /* ORDERBOOK_LADDER_H */


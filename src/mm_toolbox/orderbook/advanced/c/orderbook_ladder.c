/**
 * @file orderbook_ladder.c
 * @brief Implementation of ladder operations for orderbook level arrays.
 *
 * Provides efficient memory shifting and level insertion for managing
 * ordered price levels in the orderbook, with optimized fast paths for
 * common cases (index 0, small moves).
 */

#include "orderbook_ladder.h"
#include <string.h>

/**
 * @brief Shift levels right starting from start_index to make room for insertion.
 *
 * When at max capacity, the last element is dropped. This only shifts data;
 * the caller must update num_levels separately.
 *
 * @param data        Pointer to the ladder data containing levels and metadata.
 * @param start_index Index from which to start shifting right.
 * @note Fast paths for start_index == 0 with small move counts (most common case).
 */
void c_ladder_roll_right(OrderbookLadderData* data, uint64_t start_index) {
    uint64_t count = data->num_levels;
    uint64_t max_levels = data->max_levels;
    OrderbookLevel* levels = data->levels;

    if (start_index > count) {
        return;
    }

    uint64_t num_move = count - start_index;

    /* At max capacity: drop last element, reduce move count */
    if (count >= max_levels && num_move > 0) {
        num_move--;
    }

    if (num_move == 0) {
        return;
    }

    /* Fast path: start_index == 0 with small moves (most common) */
    if (start_index == 0 && num_move <= 4) {
        if (num_move >= 4) levels[4] = levels[3];
        if (num_move >= 3) levels[3] = levels[2];
        if (num_move >= 2) levels[2] = levels[1];
        levels[1] = levels[0];
        return;
    }

    if (start_index == 0 && num_move <= 8) {
        uint64_t i = num_move;
        while (i > 4) {
            levels[i] = levels[i - 1];
            i--;
        }
        levels[4] = levels[3];
        levels[3] = levels[2];
        levels[2] = levels[1];
        levels[1] = levels[0];
        return;
    }

    /* General case: use memmove */
    memmove(
        &levels[start_index + 1],
        &levels[start_index],
        num_move * sizeof(OrderbookLevel)
    );
}

/**
 * @brief Shift levels left starting from start_index to remove a level.
 *
 * This only shifts data; the caller must update num_levels separately.
 *
 * @param data        Pointer to the ladder data containing levels and metadata.
 * @param start_index Index from which to start shifting left.
 * @note Fast paths for start_index == 0 with small move counts (most common case).
 */
void c_ladder_roll_left(OrderbookLadderData* data, uint64_t start_index) {
    uint64_t count = data->num_levels;
    OrderbookLevel* levels = data->levels;

    if (start_index >= count) {
        return;
    }

    uint64_t num_move = count - start_index - 1;

    if (num_move == 0) {
        return;
    }

    /* Fast path: start_index == 0 with small moves (most common) */
    if (start_index == 0 && num_move <= 4) {
        levels[0] = levels[1];
        if (num_move >= 2) levels[1] = levels[2];
        if (num_move >= 3) levels[2] = levels[3];
        if (num_move >= 4) levels[3] = levels[4];
        return;
    }

    if (start_index == 0 && num_move <= 8) {
        levels[0] = levels[1];
        levels[1] = levels[2];
        levels[2] = levels[3];
        levels[3] = levels[4];
        uint64_t i = 4;
        while (i < num_move) {
            levels[i] = levels[i + 1];
            i++;
        }
        return;
    }

    /* General case: use memmove */
    memmove(
        &levels[start_index],
        &levels[start_index + 1],
        num_move * sizeof(OrderbookLevel)
    );
}

/**
 * @brief Insert a level directly at the specified index.
 * @param levels Pointer to the levels array.
 * @param index  Index at which to insert the level.
 * @param level  Pointer to the OrderbookLevel to insert.
 */
void c_ladder_insert_level(OrderbookLevel* levels, uint64_t index, const OrderbookLevel* level) {
    levels[index] = *level;
}

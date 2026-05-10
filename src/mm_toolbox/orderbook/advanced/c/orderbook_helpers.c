/**
 * @file orderbook_helpers.c
 * @brief Implementation of orderbook helper functions.
 *
 * Provides conversion utilities and sorting operations for OrderbookLevel
 * arrays, matching the behavior of the original Cython implementations.
 * Key components:
 *   - Price/tick and size/lot conversions with epsilon-aware flooring
 *   - In-place level swapping, reversing, and checking
 *   - Insertion sort optimized for nearly-sorted exchange data
 */

#include "orderbook_helpers.h"
#include <float.h>
#include <math.h>

/**
 * @brief Floor a double value with epsilon tolerance to handle floating-point errors.
 * @param value The floating-point value to floor.
 * @return The floored value as a uint64_t.
 * @note Uses DBL_EPSILON * 4.0 scaled by the absolute value to avoid precision issues near integer boundaries.
 */
static inline uint64_t floor_with_epsilon(double value) {
    if (!isfinite(value) || value < 0.0) {
        return 0;  /* Invalid values return 0 to prevent undefined behavior */
    }
    double eps = fabs(value) * DBL_EPSILON * 4.0;
    return (uint64_t)floor(value + eps);
}

/**
 * @brief Convert a price to tick units using floor division.
 * @param price     The price value to convert.
 * @param tick_size The size of one tick.
 * @return          The price expressed in ticks.
 */
uint64_t price_to_tick(double price, double tick_size) {
    if (tick_size <= 0.0 || !isfinite(price) || price < 0.0) {
        return 0;
    }
    return floor_with_epsilon(price / tick_size);
}

/**
 * @brief Convert a price to tick units using multiplication with pre-computed reciprocal.
 * Faster than price_to_tick() by avoiding division (2-3 cycles vs 10-20 cycles).
 * @param price           The price value to convert.
 * @param tick_size_recip The reciprocal of tick_size (1.0 / tick_size).
 * @return                The price expressed in ticks.
 */
uint64_t price_to_tick_fast(double price, double tick_size_recip) {
    if (tick_size_recip <= 0.0 || !isfinite(price) || price < 0.0) {
        return 0;
    }
    return floor_with_epsilon(price * tick_size_recip);
}

/**
 * @brief Convert a size to lot units using floor division.
 * @param size     The size value to convert.
 * @param lot_size The size of one lot.
 * @return         The size expressed in lots.
 */
uint64_t size_to_lot(double size, double lot_size) {
    if (lot_size <= 0.0 || !isfinite(size) || size < 0.0) {
        return 0;
    }
    return floor_with_epsilon(size / lot_size);
}

/**
 * @brief Convert a size to lot units using multiplication with pre-computed reciprocal.
 * Faster than size_to_lot() by avoiding division (2-3 cycles vs 10-20 cycles).
 * @param size           The size value to convert.
 * @param lot_size_recip The reciprocal of lot_size (1.0 / lot_size).
 * @return               The size expressed in lots.
 */
uint64_t size_to_lot_fast(double size, double lot_size_recip) {
    if (lot_size_recip <= 0.0 || !isfinite(size) || size < 0.0) {
        return 0;
    }
    return floor_with_epsilon(size * lot_size_recip);
}

/**
 * @brief Convert tick units back to a price.
 * @param tick      The tick value to convert.
 * @param tick_size The size of one tick.
 * @return          The price value.
 */
double tick_to_price(uint64_t tick, double tick_size) {
    return (double)tick * tick_size;
}

/**
 * @brief Convert lot units back to a size.
 * @param lot      The lot value to convert.
 * @param lot_size The size of one lot.
 * @return         The size value.
 */
double lot_to_size(uint64_t lot, double lot_size) {
    return (double)lot * lot_size;
}

/**
 * @brief Swap two OrderbookLevel structs in place.
 * @param a Pointer to the first level.
 * @param b Pointer to the second level.
 */
void swap_levels(OrderbookLevel* a, OrderbookLevel* b) {
    OrderbookLevel tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * @brief Reverse the order of levels in an array in place.
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 */
void reverse_levels_inplace(uint64_t num_levels, OrderbookLevel* levels) {
    if (num_levels == 0) {
        return;
    }
    uint64_t i = 0;
    uint64_t j = num_levels - 1;
    while (i < j) {
        swap_levels(&levels[i], &levels[j]);
        i++;
        j--;
    }
}

/**
 * @brief Check if levels are sorted by tick.
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @param ascending  If true, check ascending order; if false, descending.
 * @return           True if sorted in the specified order, false otherwise.
 */
bool is_sorted_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending) {
    if (num_levels < 2) {
        return true;
    }
    for (uint64_t i = 0; i < num_levels - 1; i++) {
        if (ascending) {
            if (levels[i].ticks > levels[i + 1].ticks) {
                return false;
            }
        } else {
            if (levels[i].ticks < levels[i + 1].ticks) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Insertion sort levels in ascending order by tick (internal helper).
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @note Optimal for nearly-sorted data (O(n) best case, O(n^2) worst case).
 */
static inline void insertion_sort_levels_ascending(uint64_t num_levels, OrderbookLevel* levels) {
    for (uint64_t i = 1; i < num_levels; i++) {
        OrderbookLevel key = levels[i];
        uint64_t j = i;
        while (j > 0 && levels[j - 1].ticks > key.ticks) {
            levels[j] = levels[j - 1];
            j--;
        }
        levels[j] = key;
    }
}

/**
 * @brief Insertion sort levels in descending order by tick (internal helper).
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @note Optimal for nearly-sorted data (O(n) best case, O(n^2) worst case).
 */
static inline void insertion_sort_levels_descending(uint64_t num_levels, OrderbookLevel* levels) {
    for (uint64_t i = 1; i < num_levels; i++) {
        OrderbookLevel key = levels[i];
        uint64_t j = i;
        while (j > 0 && levels[j - 1].ticks < key.ticks) {
            levels[j] = levels[j - 1];
            j--;
        }
        levels[j] = key;
    }
}

/**
 * @brief Sort levels by tick in place using insertion sort.
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @param ascending  If true, sort ascending; if false, sort descending.
 * @note Optimal for nearly-sorted data (O(n) best case, O(n^2) worst case).
 */
void insertion_sort_levels_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending) {
    if (num_levels < 2) {
        return;
    }
    if (ascending) {
        insertion_sort_levels_ascending(num_levels, levels);
    } else {
        insertion_sort_levels_descending(num_levels, levels);
    }
}

/**
 * @brief Sort levels by tick in place with smart algorithm.
 * Checks if already sorted first, then uses insertion sort if needed.
 * Best for exchange data which is typically pre-sorted or nearly-sorted.
 * @param num_levels Number of levels in the array.
 * @param levels     Pointer to the array of OrderbookLevel structs.
 * @param ascending  If true, sort ascending; if false, sort descending.
 */
void sort_levels_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending) {
    if (is_sorted_by_tick(num_levels, levels, ascending)) {
        return;
    }
    insertion_sort_levels_by_tick(num_levels, levels, ascending);
}

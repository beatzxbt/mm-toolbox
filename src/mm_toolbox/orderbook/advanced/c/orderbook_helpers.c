/**
 * orderbook_helpers.c - Implementation of orderbook helper functions.
 *
 * Provides conversion utilities and sorting operations for OrderbookLevel
 * arrays, matching the behavior of the original Cython implementations.
 */

#include "orderbook_helpers.h"
#include <float.h>
#include <math.h>

static inline uint64_t floor_with_epsilon(double value) {
    double eps = fabs(value) * DBL_EPSILON * 4.0;
    return (uint64_t)floor(value + eps);
}

uint64_t price_to_tick(double price, double tick_size) {
    return floor_with_epsilon(price / tick_size);
}

uint64_t price_to_tick_fast(double price, double tick_size_recip) {
    return floor_with_epsilon(price * tick_size_recip);
}

uint64_t size_to_lot(double size, double lot_size) {
    return floor_with_epsilon(size / lot_size);
}

uint64_t size_to_lot_fast(double size, double lot_size_recip) {
    return floor_with_epsilon(size * lot_size_recip);
}

double tick_to_price(uint64_t tick, double tick_size) {
    return (double)tick * tick_size;
}

double lot_to_size(uint64_t lot, double lot_size) {
    return (double)lot * lot_size;
}

void swap_levels(OrderbookLevel* a, OrderbookLevel* b) {
    OrderbookLevel tmp = *a;
    *a = *b;
    *b = tmp;
}

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

void sort_levels_by_tick(uint64_t num_levels, OrderbookLevel* levels, bool ascending) {
    /* Smart sort: check if already sorted, then use insertion sort if needed */
    if (is_sorted_by_tick(num_levels, levels, ascending)) {
        return;  /* Fast path: already sorted */
    }
    insertion_sort_levels_by_tick(num_levels, levels, ascending);
}

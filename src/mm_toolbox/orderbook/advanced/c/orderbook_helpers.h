/**
 * @file orderbook_helpers.h
 * @brief Conversion and normalized-entry helper functions.
 */

#ifndef ORDERBOOK_HELPERS_H
#define ORDERBOOK_HELPERS_H

#include "orderbook_types.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

void reverse_entries(uint64_t num_entries, OrderbookEntry* entries);

static inline uint64_t orderbook_floor_with_epsilon(double value) {
    if (!isfinite(value) || value < 0.0) {
        return 0;
    }
    double eps = fabs(value) * DBL_EPSILON * 4.0;
    return (uint64_t)floor(value + eps);
}

static inline uint64_t price_to_tick(double price, double tick_size_recip) {
    if (tick_size_recip <= 0.0 || !isfinite(price) || price < 0.0) {
        return 0;
    }
    return orderbook_floor_with_epsilon(price * tick_size_recip);
}

static inline uint64_t size_to_lot(double size, double lot_size_recip) {
    if (lot_size_recip <= 0.0 || !isfinite(size) || size < 0.0) {
        return 0;
    }
    return orderbook_floor_with_epsilon(size * lot_size_recip);
}

static inline double tick_to_price(uint64_t tick, double tick_size) {
    return (double)tick * tick_size;
}

static inline double lot_to_size(uint64_t lot, double lot_size) {
    return (double)lot * lot_size;
}

static inline void orderbook_entry_assign(
    OrderbookEntry* entry,
    uint64_t ticks,
    uint64_t lots,
    uint64_t norders
) {
    entry->ticks = ticks;
    entry->lots = lots;
    entry->norders = norders;
    entry->_pad = 0;
}

static inline void swap_entries(OrderbookEntry* a, OrderbookEntry* b) {
    OrderbookEntry tmp = *a;
    *a = *b;
    *b = tmp;
}

#endif /* ORDERBOOK_HELPERS_H */

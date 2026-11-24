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

/** Exact floored weighted tick for bid_ticks <= ask_ticks; zero total size gives 0. */
uint64_t orderbook_weighted_mid_tick(
    uint64_t bid_ticks,
    uint64_t ask_ticks,
    uint64_t bid_lots,
    uint64_t ask_lots
);

void reverse_entries(uint64_t num_entries, OrderbookEntry* entries);

/**
 * Compact equal tick runs in sorted input before any reversal. The final
 * original entry supplies the absolute quantity and order count. Optionally
 * omit final zero quantities when constructing a snapshot.
 */
uint64_t compact_entries(uint64_t num_entries, OrderbookEntry* entries, bool drop_zero);

static inline uint64_t orderbook_floor_with_epsilon(double value) {
    if (!isfinite(value) || value < 0.0 || value >= 18446744073709551616.0) {
        return 0;
    }
    double integer = floor(value);
    if (value == integer) {
        return (uint64_t)integer;
    }
    double eps = fmin(fabs(value) * DBL_EPSILON * 4.0, 0.5);
    if (1.0 - (value - integer) <= eps) {
        integer += 1.0;
    }
    return (uint64_t)integer;
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

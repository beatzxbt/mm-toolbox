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

/** Exact inclusive impact boundary, saturated to the unsigned tick domain. */
uint64_t orderbook_impact_limit_tick(uint64_t anchor, uint64_t bps, bool is_buy);

void reverse_entries(uint64_t num_entries, OrderbookEntry* entries);

/**
 * Compact equal tick runs in sorted input before any reversal. The final
 * original entry supplies the absolute quantity and order count. Optionally
 * omit final zero quantities when constructing a snapshot.
 */
uint64_t compact_entries(uint64_t num_entries, OrderbookEntry* entries, bool drop_zero);

static inline double orderbook_canonical_scale(double scale) {
    if (scale >= 1.0 && scale < 4503599627370496.0 && scale != floor(scale)) {
        double nearest = floor(scale + 0.5);
        if (fabs(scale - nearest) <= nearest * DBL_EPSILON) {
            uint64_t remaining = (uint64_t)nearest;
            while (remaining > 1 && remaining % 10 == 0) {
                remaining /= 10;
            }
            if (remaining == 1) {
                return nearest;
            }
        }
    }
    return scale;
}

static inline uint64_t orderbook_floor_grid(double raw, double scale) {
    scale = orderbook_canonical_scale(scale);
    double value = raw * scale;
    if (!isfinite(value) || value < 0.0 || value >= 18446744073709551616.0) {
        return 0;
    }
    double integer = floor(value);
    if (value < 4503599627370496.0) {
        double current;
        double following;
        if (scale >= 1.0) {
            current = integer / scale;
            following = (integer + 1.0) / scale;
        } else {
            double quantum = orderbook_canonical_scale(1.0 / scale);
            current = integer * quantum;
            following = (integer + 1.0) * quantum;
        }
        if (raw < current && integer > 0.0) {
            integer -= 1.0;
        } else if (raw >= following) {
            integer += 1.0;
        }
    }
    return (uint64_t)integer;
}

static inline uint64_t price_to_tick(double price, double tick_size_recip) {
    if (tick_size_recip <= 0.0 || !isfinite(tick_size_recip) || !isfinite(price) || price < 0.0) {
        return 0;
    }
    return orderbook_floor_grid(price, tick_size_recip);
}

static inline uint64_t size_to_lot(double size, double lot_size_recip) {
    if (lot_size_recip <= 0.0 || !isfinite(lot_size_recip) || !isfinite(size) || size < 0.0) {
        return 0;
    }
    return orderbook_floor_grid(size, lot_size_recip);
}

static inline double tick_to_price(uint64_t tick, double tick_size) {
    return tick_size < 1.0 ? (double)tick / orderbook_canonical_scale(1.0 / tick_size)
                           : (double)tick * tick_size;
}

static inline double lot_to_size(uint64_t lot, double lot_size) {
    return tick_to_price(lot, lot_size);
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

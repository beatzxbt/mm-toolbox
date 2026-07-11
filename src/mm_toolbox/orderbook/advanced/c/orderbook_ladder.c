/**
 * @file orderbook_ladder.c
 * @brief Ring-backed normalized orderbook ladder operations.
 */

#include "orderbook_ladder.h"

static inline uint64_t physical_index(OrderbookLadderData* data, uint64_t index) {
    return (data->head + index) % data->max_levels;
}

static inline int entry_before(uint64_t left_ticks, uint64_t right_ticks, int ascending) {
    return ascending ? (left_ticks < right_ticks) : (left_ticks > right_ticks);
}

OrderbookEntry* c_ladder_at(OrderbookLadderData* data, uint64_t index) {
    return &data->levels[physical_index(data, index)];
}

OrderbookEntry* c_ladder_top(OrderbookLadderData* data) {
    return c_ladder_at(data, 0);
}

OrderbookEntry* c_ladder_bottom(OrderbookLadderData* data) {
    return c_ladder_at(data, data->num_levels - 1);
}

void c_ladder_insert_entry(OrderbookLadderData* data, uint64_t index, const OrderbookEntry* entry) {
    *c_ladder_at(data, index) = *entry;
}

void c_ladder_roll_right(OrderbookLadderData* data, uint64_t start_index) {
    uint64_t count = data->num_levels;
    if (data->max_levels == 0 || start_index > count) {
        return;
    }
    if (count == 0) {
        return;
    }

    uint64_t limit = count < data->max_levels ? count : data->max_levels - 1;

    if (start_index <= count - start_index) {
        data->head = (data->head + data->max_levels - 1) % data->max_levels;
        for (uint64_t i = 0; i < start_index; i++) {
            *c_ladder_at(data, i) = *c_ladder_at(data, i + 1);
        }
    } else {
        for (uint64_t i = limit; i > start_index; i--) {
            *c_ladder_at(data, i) = *c_ladder_at(data, i - 1);
        }
    }
}

void c_ladder_roll_left(OrderbookLadderData* data, uint64_t start_index) {
    uint64_t count = data->num_levels;
    if (data->max_levels == 0 || start_index >= count) {
        return;
    }
    if (count <= 1) {
        return;
    }

    if (start_index <= count - start_index - 1) {
        for (uint64_t i = start_index; i > 0; i--) {
            *c_ladder_at(data, i) = *c_ladder_at(data, i - 1);
        }
        data->head = (data->head + 1) % data->max_levels;
    } else {
        for (uint64_t i = start_index; i < count - 1; i++) {
            *c_ladder_at(data, i) = *c_ladder_at(data, i + 1);
        }
    }
}

void c_ladder_export_levels(OrderbookLevel* out, OrderbookLadderData* data, double tick_size, double lot_size) {
    for (uint64_t i = 0; i < data->num_levels; i++) {
        OrderbookEntry* entry = c_ladder_at(data, i);
        out[i].price = tick_to_price(entry->ticks, tick_size);
        out[i].size = lot_to_size(entry->lots, lot_size);
        out[i].norders = entry->norders;
    }
}

uint64_t c_ladder_ask_lower_bound(OrderbookLadderData* data, uint64_t ticks, uint64_t start, uint64_t end) {
    uint64_t lo = start;
    uint64_t hi = end;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) / 2);
        if (c_ladder_at(data, mid)->ticks < ticks) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

uint64_t c_ladder_bid_lower_bound(OrderbookLadderData* data, uint64_t ticks, uint64_t start, uint64_t end) {
    uint64_t lo = start;
    uint64_t hi = end;
    while (lo < hi) {
        uint64_t mid = lo + ((hi - lo) / 2);
        if (c_ladder_at(data, mid)->ticks > ticks) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

uint64_t c_ladder_ask_seek_start(OrderbookLadderData* data, uint64_t ticks) {
    return c_ladder_ask_lower_bound(data, ticks, 0, data->num_levels);
}

uint64_t c_ladder_bid_seek_start(OrderbookLadderData* data, uint64_t ticks) {
    return c_ladder_bid_lower_bound(data, ticks, 0, data->num_levels);
}

void c_ladder_apply_sorted_deltas(
    OrderbookLadderData* data,
    OrderbookEntry* updates,
    uint64_t update_count,
    OrderbookEntry* scratch
) {
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t out = 0;
    uint64_t count = data->num_levels;
    uint64_t max_levels = data->max_levels;
    int ascending = data->is_price_ascending;

    while (out < max_levels && (i < count || j < update_count)) {
        if (i >= count) {
            if (updates[j].lots != 0) {
                scratch[out++] = updates[j];
            }
            j++;
            continue;
        }
        if (j >= update_count) {
            scratch[out++] = *c_ladder_at(data, i++);
            continue;
        }

        OrderbookEntry current = *c_ladder_at(data, i);
        OrderbookEntry update = updates[j];
        if (current.ticks == update.ticks) {
            if (update.lots != 0) {
                scratch[out++] = update;
            }
            i++;
            j++;
        } else if (entry_before(update.ticks, current.ticks, ascending)) {
            if (update.lots != 0) {
                scratch[out++] = update;
            }
            j++;
        } else {
            scratch[out++] = current;
            i++;
        }
    }

    data->head = 0;
    data->num_levels = out;
    for (uint64_t k = 0; k < out; k++) {
        data->levels[k] = scratch[k];
    }
}

/**
 * @file orderbook_ladder.h
 * @brief Ring-backed normalized orderbook ladder operations.
 */

#ifndef ORDERBOOK_LADDER_H
#define ORDERBOOK_LADDER_H

#include "orderbook_helpers.h"
#include "orderbook_types.h"
#include <stdint.h>

typedef struct {
    uint64_t num_levels;
    uint64_t max_levels;
    OrderbookEntry* levels;
    uint64_t head;
    int is_price_ascending;
} OrderbookLadderData;

OrderbookEntry* c_ladder_at(OrderbookLadderData* data, uint64_t index);
OrderbookEntry* c_ladder_top(OrderbookLadderData* data);
OrderbookEntry* c_ladder_bottom(OrderbookLadderData* data);

void c_ladder_roll_right(OrderbookLadderData* data, uint64_t start_index);
void c_ladder_roll_left(OrderbookLadderData* data, uint64_t start_index);
void c_ladder_insert_entry(OrderbookLadderData* data, uint64_t index, const OrderbookEntry* entry);
void c_ladder_export_levels(OrderbookLevel* out, OrderbookLadderData* data, double tick_size, double lot_size);

uint64_t c_ladder_ask_lower_bound(OrderbookLadderData* data, uint64_t ticks, uint64_t start, uint64_t end);
uint64_t c_ladder_bid_lower_bound(OrderbookLadderData* data, uint64_t ticks, uint64_t start, uint64_t end);
uint64_t c_ladder_ask_seek_start(OrderbookLadderData* data, uint64_t ticks);
uint64_t c_ladder_bid_seek_start(OrderbookLadderData* data, uint64_t ticks);

void c_ladder_apply_sorted_deltas(
    OrderbookLadderData* data,
    OrderbookEntry* updates,
    uint64_t update_count,
    OrderbookEntry* scratch
);

#endif /* ORDERBOOK_LADDER_H */

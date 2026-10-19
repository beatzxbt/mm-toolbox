/**
 * @file orderbook_helpers.c
 * @brief Conversion and normalized-entry helper implementations.
 */

#include "orderbook_helpers.h"

uint64_t compact_entries(uint64_t num_entries, OrderbookEntry* entries, bool drop_zero) {
    uint64_t out = 0;
    uint64_t i = 0;
    while (i < num_entries) {
        uint64_t last = i;
        while (last + 1 < num_entries && entries[last + 1].ticks == entries[i].ticks) {
            last++;
        }
        if (!drop_zero || entries[last].lots != 0) {
            entries[out++] = entries[last];
        }
        i = last + 1;
    }
    return out;
}

void reverse_entries(uint64_t num_entries, OrderbookEntry* entries) {
    if (num_entries < 2) {
        return;
    }
    uint64_t i = 0;
    uint64_t j = num_entries - 1;
    while (i < j) {
        swap_entries(&entries[i], &entries[j]);
        i++;
        j--;
    }
}

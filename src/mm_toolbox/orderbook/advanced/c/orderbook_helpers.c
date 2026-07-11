/**
 * @file orderbook_helpers.c
 * @brief Conversion and normalized-entry helper implementations.
 */

#include "orderbook_helpers.h"

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

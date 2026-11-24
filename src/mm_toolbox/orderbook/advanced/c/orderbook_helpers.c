/**
 * @file orderbook_helpers.c
 * @brief Conversion and normalized-entry helper implementations.
 */

#include "orderbook_helpers.h"

#if !defined(__SIZEOF_INT128__) || defined(MM_TOOLBOX_FORCE_PORTABLE_WMID)
/* 32-bit limbs give the exact high/low halves without wider integer types. */
static void multiply_u64_wide(uint64_t a, uint64_t b, uint64_t* high, uint64_t* low) {
    const uint64_t mask = UINT64_C(0xffffffff);
    uint64_t a_low = a & mask;
    uint64_t a_high = a >> 32;
    uint64_t b_low = b & mask;
    uint64_t b_high = b >> 32;
    uint64_t first = a_low * b_low;
    uint64_t middle = a_high * b_low + (first >> 32);
    uint64_t upper = middle >> 32;
    middle = (middle & mask) + a_low * b_high;
    *high = a_high * b_high + upper + (middle >> 32);
    *low = (middle << 32) | (first & mask);
}

/* The caller proves the quotient fits in 64 bits, so numerator_high is already
 * smaller than the 65-bit divisor. Process only the remaining 64 numerator bits.
 * A shifted remainder needs at most 66 bits; its high limb stays in [0, 3]. */
static uint64_t divide_bounded_u128(
    uint64_t numerator_high,
    uint64_t numerator_low,
    uint64_t denominator_high,
    uint64_t denominator_low
) {
    uint64_t remainder_high = 0;
    uint64_t remainder_low = numerator_high;
    uint64_t quotient = 0;
    for (int bit = 63; bit >= 0; --bit) {
        remainder_high = (remainder_high << 1) | (remainder_low >> 63);
        remainder_low = (remainder_low << 1) | ((numerator_low >> bit) & UINT64_C(1));
        if (remainder_high > denominator_high ||
            (remainder_high == denominator_high && remainder_low >= denominator_low)) {
            uint64_t previous_low = remainder_low;
            remainder_low -= denominator_low;
            remainder_high -= denominator_high + (previous_low < denominator_low);
            quotient |= UINT64_C(1) << bit;
        }
    }
    return quotient;
}
#endif

uint64_t orderbook_weighted_mid_tick(
    uint64_t bid_ticks,
    uint64_t ask_ticks,
    uint64_t bid_lots,
    uint64_t ask_lots
) {
    if (ask_lots == 0) {
        return bid_lots == 0 ? 0 : bid_ticks;
    }
    if (bid_lots == 0) {
        return ask_ticks;
    }
    if (ask_ticks <= bid_ticks) {
        return bid_ticks;
    }

    uint64_t distance = ask_ticks - bid_ticks;
    /* Unsigned addition wraps deliberately here; the carry preserves bit 64. */
    uint64_t total_low = bid_lots + ask_lots;
    uint64_t total_high = total_low < bid_lots;
    uint64_t offset;
#if defined(__SIZEOF_INT128__) && !defined(MM_TOOLBOX_FORCE_PORTABLE_WMID)
    /* distance*ask_lots < 2^128 and total size < 2^65. Their quotient is
     * bounded by distance, so adding it to bid_ticks cannot exceed ask_ticks. */
    __uint128_t numerator = (__uint128_t)distance * (__uint128_t)ask_lots;
    if (total_high == 0 && (numerator >> 64) == 0) {
        offset = (uint64_t)numerator / total_low;
    } else {
        __uint128_t denominator = ((__uint128_t)total_high << 64) | total_low;
        offset = (uint64_t)(numerator / denominator);
    }
#else
    uint64_t numerator_high;
    uint64_t numerator_low;
    multiply_u64_wide(distance, ask_lots, &numerator_high, &numerator_low);
    if (total_high == 0 && numerator_high == 0) {
        offset = numerator_low / total_low;
    } else {
        offset = divide_bounded_u128(numerator_high, numerator_low, total_high, total_low);
    }
#endif
    return bid_ticks + offset;
}

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

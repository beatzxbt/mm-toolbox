/**
 * shm_helpers.c - Implementation of SHM helper functions.
 *
 * Optimized memory operations for ring buffer with alignment-aware paths
 * and efficient wrap-around handling using split memcpy.
 */

#include "shm_helpers.h"
#include <string.h>

void shm_write_u64_le(unsigned char* base, uint64_t idx, uint64_t mask, uint64_t val) {
    uint64_t pos = idx & mask;
    uint64_t capacity = mask + 1;

    if (pos + 8 <= capacity) {
        memcpy(base + pos, &val, 8);
    } else {
        /* Wrap-around case - byte-by-byte with masking */
        base[(idx + 0) & mask] = (unsigned char)(val & 0xFF);
        base[(idx + 1) & mask] = (unsigned char)((val >> 8) & 0xFF);
        base[(idx + 2) & mask] = (unsigned char)((val >> 16) & 0xFF);
        base[(idx + 3) & mask] = (unsigned char)((val >> 24) & 0xFF);
        base[(idx + 4) & mask] = (unsigned char)((val >> 32) & 0xFF);
        base[(idx + 5) & mask] = (unsigned char)((val >> 40) & 0xFF);
        base[(idx + 6) & mask] = (unsigned char)((val >> 48) & 0xFF);
        base[(idx + 7) & mask] = (unsigned char)((val >> 56) & 0xFF);
    }
}

uint64_t shm_read_u64_le(const unsigned char* base, uint64_t idx, uint64_t mask) {
    uint64_t pos = idx & mask;
    uint64_t capacity = mask + 1;

    if (pos + 8 <= capacity) {
        uint64_t val;
        memcpy(&val, base + pos, 8);
        return val;
    } else {
        /* Wrap-around case - byte-by-byte with masking */
        return ((uint64_t)base[(idx + 0) & mask])
             | ((uint64_t)base[(idx + 1) & mask] << 8)
             | ((uint64_t)base[(idx + 2) & mask] << 16)
             | ((uint64_t)base[(idx + 3) & mask] << 24)
             | ((uint64_t)base[(idx + 4) & mask] << 32)
             | ((uint64_t)base[(idx + 5) & mask] << 40)
             | ((uint64_t)base[(idx + 6) & mask] << 48)
             | ((uint64_t)base[(idx + 7) & mask] << 56);
    }
}

void shm_copy_into_ring(unsigned char* ring, uint64_t start, uint64_t mask,
                        const unsigned char* src, size_t n, uint64_t capacity) {
    if (n == 0) return;

    uint64_t idx = start & mask;
    uint64_t endspace = capacity - idx;

    if (n <= endspace) {
        /* No wrap - single memcpy */
        memcpy(ring + idx, src, n);
    } else {
        /* Wrap - two memcpy calls */
        memcpy(ring + idx, src, (size_t)endspace);
        memcpy(ring, src + (size_t)endspace, n - (size_t)endspace);
    }
}

void shm_copy_from_ring(unsigned char* dst, const unsigned char* ring, uint64_t start,
                        uint64_t mask, size_t n, uint64_t capacity) {
    if (n == 0) return;

    uint64_t idx = start & mask;
    uint64_t endspace = capacity - idx;

    if (n <= endspace) {
        /* No wrap - single memcpy */
        memcpy(dst, ring + idx, n);
    } else {
        /* Wrap - two memcpy calls */
        memcpy(dst, ring + idx, (size_t)endspace);
        memcpy(dst + (size_t)endspace, ring, n - (size_t)endspace);
    }
}

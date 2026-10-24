#include "shm_helpers.h"
#include <string.h>

#ifdef SHM_TEST_AFTER_READ_WORD
extern void SHM_TEST_AFTER_READ_WORD(void);
#endif

void shm_copy_into_ring(unsigned char* ring, uint64_t start, uint64_t mask,
                        const unsigned char* src, size_t n, uint64_t capacity) {
    if (capacity < 8) return;
    while (n) {
        uint64_t idx = start & mask;
        size_t offset = (size_t)(idx & 7);
        size_t take = 8 - offset;
        uint64_t word = 0;
        uint64_t* slot = (uint64_t*)(ring + (idx & ~7ULL));
        if (take > n) take = n;
        if (offset || take < 8) word = __atomic_load_n(slot, __ATOMIC_RELAXED);
        memcpy((unsigned char*)&word + offset, src, take);
        __atomic_store_n(slot, word, __ATOMIC_RELEASE);
        src += take;
        start += take;
        n -= take;
    }
}

void shm_copy_from_ring(unsigned char* dst, const unsigned char* ring, uint64_t start,
                        uint64_t mask, size_t n, uint64_t capacity) {
    if (capacity < 8) return;
    if ((start & 7) && n) {
        size_t offset = (size_t)(start & 7);
        size_t take = 8 - offset;
        if (take > n) take = n;
        uint64_t word = __atomic_load_n(
            (const uint64_t*)(ring + ((start & mask) & ~7ULL)), __ATOMIC_RELAXED);
#ifdef SHM_TEST_AFTER_READ_WORD
        SHM_TEST_AFTER_READ_WORD();
#endif
        memcpy(dst, (const unsigned char*)&word + offset, take);
        start += take; dst += take; n -= take;
    }
    while (n >= 8) {
        uint64_t word = __atomic_load_n(
            (const uint64_t*)(ring + (start & mask)), __ATOMIC_RELAXED);
#ifdef SHM_TEST_AFTER_READ_WORD
        SHM_TEST_AFTER_READ_WORD();
#endif
        memcpy(dst, &word, 8);
        start += 8; dst += 8; n -= 8;
    }
    if (n) {
        uint64_t word = __atomic_load_n(
            (const uint64_t*)(ring + (start & mask)), __ATOMIC_RELAXED);
#ifdef SHM_TEST_AFTER_READ_WORD
        SHM_TEST_AFTER_READ_WORD();
#endif
        memcpy(dst, &word, n);
    }
}

void shm_write_u64_le(unsigned char* base, uint64_t idx, uint64_t mask, uint64_t val) {
    if (!(idx & 7) && mask >= 7) {
        __atomic_store_n((uint64_t*)(base + (idx & mask)), val, __ATOMIC_RELEASE);
        return;
    }
    shm_copy_into_ring(base, idx, mask, (const unsigned char*)&val, 8, mask + 1);
}

uint64_t shm_read_u64_le(const unsigned char* base, uint64_t idx, uint64_t mask) {
    uint64_t val = 0;
    if (!(idx & 7) && mask >= 7) {
        val = __atomic_load_n((const uint64_t*)(base + (idx & mask)), __ATOMIC_RELAXED);
#ifdef SHM_TEST_AFTER_READ_WORD
        SHM_TEST_AFTER_READ_WORD();
#endif
        return val;
    }
    shm_copy_from_ring((unsigned char*)&val, base, idx, mask, 8, mask + 1);
    return val;
}

/* Production payloads start aligned. Padding is part of the record and uses
 * a zeroed private word; no read extends beyond the caller's payload. */
void shm_copy_payload_into_ring(unsigned char* ring, uint64_t start, uint64_t mask,
                                const unsigned char* src, size_t n) {
    while (n >= 8) {
        uint64_t word;
        memcpy(&word, src, 8);
        __atomic_store_n((uint64_t*)(ring + (start & mask)), word, __ATOMIC_RELEASE);
        src += 8; start += 8; n -= 8;
    }
    if (n) {
        uint64_t word = 0;
        memcpy(&word, src, n);
        __atomic_store_n((uint64_t*)(ring + (start & mask)), word, __ATOMIC_RELEASE);
    }
}

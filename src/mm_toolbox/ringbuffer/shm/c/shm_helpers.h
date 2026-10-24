#ifndef SHM_HELPERS_H
#define SHM_HELPERS_H
#include "shm_types.h"
#include <stddef.h>

/* Ring storage is always accessed as aligned uint64_t atomic words. Payload
 * stores release; speculative reads relaxed, followed by snapshot validation
 * or retirement's acquire fence. Private source/destination copies use memcpy.
 * The generic offset helpers also support unaligned offsets within a ring of
 * at least eight bytes; production record boundaries are always aligned. */
void shm_write_u64_le(unsigned char* base, uint64_t idx, uint64_t mask, uint64_t val);
uint64_t shm_read_u64_le(const unsigned char* base, uint64_t idx, uint64_t mask);
void shm_copy_into_ring(unsigned char* ring, uint64_t start, uint64_t mask,
                        const unsigned char* src, size_t n, uint64_t capacity);
void shm_copy_payload_into_ring(unsigned char* ring, uint64_t start, uint64_t mask,
                                const unsigned char* src, size_t n);
void shm_copy_from_ring(unsigned char* dst, const unsigned char* ring, uint64_t start,
                        uint64_t mask, size_t n, uint64_t capacity);
#endif

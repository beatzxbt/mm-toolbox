/**
 * shm_helpers.h - Helper functions for SHM ring buffer memory operations.
 *
 * Provides low-level utilities for message encoding/decoding and ring buffer
 * memory operations with wrap-around handling. All functions are designed for
 * use in nogil contexts.
 */

#ifndef SHM_HELPERS_H
#define SHM_HELPERS_H

#include "shm_types.h"
#include <stddef.h>

/**
 * Write 64-bit value in little-endian format with wrap-around handling.
 *
 * Handles the case where the 8-byte value may span the ring buffer boundary.
 * Uses direct assignment for aligned writes and byte-by-byte for edge cases.
 *
 * @param base  Ring buffer data pointer.
 * @param idx   Logical index (may exceed capacity, will be masked).
 * @param mask  Capacity - 1 for fast modulo.
 * @param val   Value to write.
 */
void shm_write_u64_le(unsigned char* base, uint64_t idx, uint64_t mask, uint64_t val);

/**
 * Read 64-bit value in little-endian format with wrap-around handling.
 *
 * Handles the case where the 8-byte value may span the ring buffer boundary.
 * Uses direct assignment for aligned reads and byte-by-byte for edge cases.
 *
 * @param base  Ring buffer data pointer.
 * @param idx   Logical index (may exceed capacity, will be masked).
 * @param mask  Capacity - 1 for fast modulo.
 * @return      The decoded 64-bit value.
 */
uint64_t shm_read_u64_le(const unsigned char* base, uint64_t idx, uint64_t mask);

/**
 * Copy contiguous bytes into ring buffer with wrap-around handling.
 *
 * Splits the copy into two memcpy calls if the data spans the buffer boundary.
 *
 * @param ring      Ring buffer data pointer.
 * @param start     Starting logical index.
 * @param mask      Capacity - 1.
 * @param src       Source data pointer.
 * @param n         Number of bytes to copy.
 * @param capacity  Ring buffer capacity.
 */
void shm_copy_into_ring(unsigned char* ring, uint64_t start, uint64_t mask,
                        const unsigned char* src, size_t n, uint64_t capacity);

/**
 * Copy contiguous bytes from ring buffer with wrap-around handling.
 *
 * Splits the copy into two memcpy calls if the data spans the buffer boundary.
 *
 * @param dst       Destination pointer.
 * @param ring      Ring buffer data pointer.
 * @param start     Starting logical index.
 * @param mask      Capacity - 1.
 * @param n         Number of bytes to copy.
 * @param capacity  Ring buffer capacity.
 */
void shm_copy_from_ring(unsigned char* dst, const unsigned char* ring, uint64_t start,
                        uint64_t mask, size_t n, uint64_t capacity);

#endif /* SHM_HELPERS_H */

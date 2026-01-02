/**
 * shm_types.h - Type definitions for SHM ring buffer.
 *
 * Defines the shared memory header structure and message layout constants
 * for lock-free SPSC (Single Producer, Single Consumer) communication.
 * All structs are designed for 64-byte cache line alignment.
 */

#ifndef SHM_TYPES_H
#define SHM_TYPES_H

#include <stdint.h>

/* Message format: 8-byte little-endian length + payload */
#define SHM_MSG_HEADER_SIZE 8

/* Magic value for header validation: 'SHBR' */
#define SHM_MAGIC 0x53484252ULL

/**
 * ShmHeader - 64-byte aligned shared memory header for ring buffer.
 *
 * All fields are 64-bit unsigned integers for atomic operations and cache
 * alignment. This struct matches the Cython definition in header.pxd exactly.
 *
 * Fields:
 *   magic                   - Magic value 0x53484252 ('SHBR') for validation.
 *   capacity                - Ring buffer capacity in bytes (power of 2).
 *   mask                    - Capacity - 1, used for fast modulo via bitwise AND.
 *   write_pos               - Current write position (monotonically increasing).
 *   read_pos                - Current read position (monotonically increasing).
 *   msg_count               - Number of messages currently in the buffer.
 *   latest_insert_time_ns   - Timestamp of most recent insert (monotonic ns).
 *   latest_consume_time_ns  - Timestamp of most recent consume (monotonic ns).
 */
typedef struct {
    uint64_t magic;
    uint64_t capacity;
    uint64_t mask;
    uint64_t write_pos;
    uint64_t read_pos;
    uint64_t msg_count;
    uint64_t latest_insert_time_ns;
    uint64_t latest_consume_time_ns;
} ShmHeader;  /* 64 bytes total - cache line aligned */

#endif /* SHM_TYPES_H */

/**
 * @file shm_types.h
 * @brief Type definitions for SHM ring buffer.
 *
 * Defines the shared memory header structure and message layout constants
 * for lock-free SPSC (Single Producer, Single Consumer) communication.
 * All structs are designed for 64-byte cache line alignment.
 */

#ifndef SHM_TYPES_H
#define SHM_TYPES_H

#include <stdint.h>

/**
 * @brief Message format: 8-byte little-endian length + payload.
 */
#define SHM_MSG_HEADER_SIZE 8

/**
 * @brief Magic value for header validation: 'SHBR'.
 */
#define SHM_MAGIC 0x53484252ULL

/**
 * @brief 64-byte aligned shared memory header for ring buffer.
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
} ShmHeader;

/**
 * @brief MPSC magic value: 'SHMP' (Shared Memory Multi-Producer).
 */
#define SHM_MPSC_MAGIC 0x53484D50ULL

/**
 * @brief Global header for MPSC sharded ring buffer.
 *
 * Located at offset 0 of the shared memory region. Manages the number of
 * sub-rings and provides an atomic counter for producer ring selection.
 *
 * Fields:
 *   magic              - Magic value 0x53484D50 ('SHMP') for validation.
 *   num_rings          - Number of sub-rings (shards).
 *   ring_capacity      - Capacity of each individual sub-ring in bytes.
 *   next_producer_ring - Atomic counter for round-robin producer assignment.
 *   reserved           - Padding to 64 bytes.
 */
typedef struct {
    uint64_t magic;
    uint64_t num_rings;
    uint64_t ring_capacity;
    uint64_t next_producer_ring;
    uint64_t reserved[4];
} ShmMpscGlobalHeader;

/**
 * @brief Per-sub-ring header for MPSC architecture.
 *
 * Each sub-ring uses this header instead of ShmHeader. The first three
 * slots are reserved padding so that the offsets of write_pos, read_pos,
 * msg_count, and the timestamp fields match ShmHeader exactly. This allows
 * the existing shm_producer_insert / shm_consumer_consume C functions to
 * operate on a ShmSubRingHeader* safely when it is cast to ShmHeader*.
 *
 * Fields:
 *   _pad0                    - offset 0  - reserved (was magic).
 *   _pad1                    - offset 8  - reserved (was capacity).
 *   _pad2                    - offset 16 - reserved (was mask).
 *   write_pos                - offset 24 - Current write position.
 *   read_pos                 - offset 32 - Current read position.
 *   msg_count                - offset 40 - Number of messages in this sub-ring.
 *   latest_insert_time_ns    - offset 48 - Timestamp of most recent insert.
 *   latest_consume_time_ns   - offset 56 - Timestamp of most recent consume.
 */
typedef struct {
    uint64_t _pad0;
    uint64_t _pad1;
    uint64_t _pad2;
    uint64_t write_pos;
    uint64_t read_pos;
    uint64_t msg_count;
    uint64_t latest_insert_time_ns;
    uint64_t latest_consume_time_ns;
} ShmSubRingHeader;

#endif /* SHM_TYPES_H */

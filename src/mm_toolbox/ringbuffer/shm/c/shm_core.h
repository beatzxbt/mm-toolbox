/**
 * shm_core.h - Core insert/consume operations for SHM ring buffer.
 *
 * Provides high-performance, lock-free producer and consumer operations
 * with integrated timestamping using monotonic time. All functions are
 * designed for use in nogil contexts with atomic memory ordering.
 */

#ifndef SHM_CORE_H
#define SHM_CORE_H

#include "shm_types.h"
#include <stddef.h>
#include <stdint.h>

/**
 * ShmProducerContext - Context for producer operations (cached state).
 *
 * Maintains local copies of frequently accessed values to reduce
 * atomic operations on the shared header.
 *
 * Fields:
 *   hdr          - Pointer to shared memory header.
 *   data         - Pointer to ring buffer data region.
 *   capacity     - Buffer capacity in bytes.
 *   mask         - Capacity - 1.
 *   cached_read  - Last known read position (for producer optimization).
 *   cached_write - Current write position (local cache).
 */
typedef struct {
    ShmHeader* hdr;
    unsigned char* data;
    uint64_t capacity;
    uint64_t mask;
    uint64_t cached_read;
    uint64_t cached_write;
} ShmProducerContext;

/**
 * ShmConsumerContext - Context for consumer operations (cached state).
 *
 * Fields:
 *   hdr       - Pointer to shared memory header.
 *   data      - Pointer to ring buffer data region.
 *   capacity  - Buffer capacity in bytes.
 *   mask      - Capacity - 1.
 *   spin_wait - Number of spin iterations before sleeping.
 */
typedef struct {
    ShmHeader* hdr;
    unsigned char* data;
    uint64_t capacity;
    uint64_t mask;
    int spin_wait;
} ShmConsumerContext;

/**
 * Insert a message into the ring buffer (producer-side).
 *
 * This function combines reserve, write, and commit operations with
 * monotonic timestamping, all in C for maximum performance. If the buffer
 * is full, oldest messages are dropped to make room.
 *
 * @param ctx          Producer context.
 * @param payload      Payload data pointer.
 * @param payload_len  Payload length in bytes.
 * @param dropped_out  Output: number of dropped messages (if any).
 * @return             1 on success, 0 on failure (message too large).
 */
int shm_producer_insert(ShmProducerContext* ctx, const unsigned char* payload,
                        size_t payload_len, uint64_t* dropped_out);

/**
 * Check if a complete message is available (peek without consuming).
 *
 * Uses atomic acquire semantics to read producer's write position.
 *
 * @param ctx          Consumer context.
 * @param msg_len_out  Output: message length if available.
 * @param read_pos_out Output: read position if available.
 * @return             1 if message available, 0 otherwise.
 */
int shm_consumer_peek_available(ShmConsumerContext* ctx, uint64_t* msg_len_out,
                                uint64_t* read_pos_out);

/**
 * Consume a message from the ring buffer (consumer-side).
 *
 * Copies message data to destination buffer and commits the read position
 * with monotonic timestamping.
 *
 * @param ctx       Consumer context.
 * @param dst       Destination buffer (must be at least msg_len bytes).
 * @param msg_len   Message length (from peek_available).
 * @param read_pos  Read position (from peek_available).
 * @return          1 on success, 0 on failure.
 */
int shm_consumer_consume(ShmConsumerContext* ctx, unsigned char* dst,
                         uint64_t msg_len, uint64_t read_pos);

#endif /* SHM_CORE_H */

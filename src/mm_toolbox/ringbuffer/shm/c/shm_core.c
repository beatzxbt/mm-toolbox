/**
 * shm_core.c - Implementation of core SHM operations.
 *
 * High-performance producer/consumer operations with atomic synchronization
 * and integrated monotonic timestamping. Uses GCC atomic builtins for
 * lock-free memory ordering.
 */

#include "shm_core.h"
#include "shm_helpers.h"
#include <time.h>

/* Atomic operation memory orders (GCC builtin values) */
#define ATOMIC_ACQUIRE 2
#define ATOMIC_RELEASE 3
#define ATOMIC_ACQ_REL 4

/* Clock constant for monotonic timing */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

/**
 * Get monotonic time in nanoseconds using clock_gettime.
 *
 * @return Monotonic nanoseconds, or 0 on error.
 */
static inline uint64_t get_monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int shm_producer_insert(ShmProducerContext* ctx, const unsigned char* payload,
                        size_t payload_len, uint64_t* dropped_out) {
    uint64_t capacity = ctx->capacity;
    uint64_t mask = ctx->mask;
    uint64_t write_pos = ctx->cached_write;
    uint64_t read_pos;
    uint64_t free_bytes;
    size_t total_len = SHM_MSG_HEADER_SIZE + payload_len;
    uint64_t msg_len;
    uint64_t dropped_pos;
    uint64_t now_ns;

    *dropped_out = 0;

    /* Sanity check: message must fit in buffer */
    if (total_len > capacity) {
        return 0;
    }

    /* Load current read position with acquire semantics */
    read_pos = __atomic_load_n(&ctx->hdr->read_pos, ATOMIC_ACQUIRE);
    ctx->cached_read = read_pos;

    free_bytes = capacity - (write_pos - read_pos);

    /* If not enough space, drop oldest messages until we have room */
    if (free_bytes < total_len) {
        dropped_pos = read_pos;
        while (1) {
            /* Read message length at current read position */
            msg_len = shm_read_u64_le(ctx->data, dropped_pos & mask, mask);

            /* Sanity check for corruption */
            if (msg_len > capacity || (dropped_pos + SHM_MSG_HEADER_SIZE + msg_len) < dropped_pos) {
                return 0;
            }

            dropped_pos += SHM_MSG_HEADER_SIZE + msg_len;
            (*dropped_out)++;
            free_bytes = capacity - (write_pos - dropped_pos);

            if (free_bytes >= total_len) {
                /* Update read position atomically and decrement message count */
                __atomic_store_n(&ctx->hdr->read_pos, dropped_pos, ATOMIC_RELEASE);
                if (*dropped_out > 0) {
                    __atomic_sub_fetch(&ctx->hdr->msg_count, *dropped_out, ATOMIC_ACQ_REL);
                }
                ctx->cached_read = dropped_pos;
                break;
            }
        }
    }

    /* Write message: length header + payload */
    shm_write_u64_le(ctx->data, write_pos & mask, mask, (uint64_t)payload_len);
    shm_copy_into_ring(ctx->data, write_pos + SHM_MSG_HEADER_SIZE, mask,
                       payload, payload_len, capacity);
    write_pos += total_len;

    /* Get monotonic timestamp */
    now_ns = get_monotonic_ns();

    /* Commit: update write_pos, msg_count, timestamp atomically */
    __atomic_store_n(&ctx->hdr->write_pos, write_pos, ATOMIC_RELEASE);
    __atomic_add_fetch(&ctx->hdr->msg_count, 1, ATOMIC_ACQ_REL);
    __atomic_store_n(&ctx->hdr->latest_insert_time_ns, now_ns, ATOMIC_RELEASE);

    ctx->cached_write = write_pos;
    return 1;
}

int shm_consumer_peek_available(ShmConsumerContext* ctx, uint64_t* msg_len_out,
                                uint64_t* read_pos_out) {
    uint64_t read_pos = __atomic_load_n(&ctx->hdr->read_pos, ATOMIC_ACQUIRE);
    uint64_t write_pos = __atomic_load_n(&ctx->hdr->write_pos, ATOMIC_ACQUIRE);
    uint64_t avail = write_pos - read_pos;
    uint64_t msg_len;

    /* Need at least 8 bytes for message length header */
    if (avail < SHM_MSG_HEADER_SIZE) {
        return 0;
    }

    /* Read message length */
    msg_len = shm_read_u64_le(ctx->data, read_pos & ctx->mask, ctx->mask);

    /* Check if complete message is available */
    if (avail < SHM_MSG_HEADER_SIZE + msg_len) {
        return 0;
    }

    *msg_len_out = msg_len;
    *read_pos_out = read_pos;
    return 1;
}

int shm_consumer_consume(ShmConsumerContext* ctx, unsigned char* dst,
                         uint64_t msg_len, uint64_t read_pos) {
    uint64_t now_ns;

    /* Copy payload from ring buffer */
    shm_copy_from_ring(dst, ctx->data, read_pos + SHM_MSG_HEADER_SIZE,
                       ctx->mask, (size_t)msg_len, ctx->capacity);

    /* Advance read position past this message */
    read_pos += SHM_MSG_HEADER_SIZE + msg_len;

    /* Get monotonic timestamp */
    now_ns = get_monotonic_ns();

    /* Commit: update read_pos, msg_count, timestamp atomically */
    __atomic_store_n(&ctx->hdr->read_pos, read_pos, ATOMIC_RELEASE);
    __atomic_sub_fetch(&ctx->hdr->msg_count, 1, ATOMIC_ACQ_REL);
    __atomic_store_n(&ctx->hdr->latest_consume_time_ns, now_ns, ATOMIC_RELEASE);

    return 1;
}

#include "shm_core.h"
#include "shm_helpers.h"
#include "../../../time/ctime_impl.h"

uint64_t shm_record_span(uint64_t len) {
    if (len > UINT64_MAX - 15) return 0;
    return (len + 15) & ~7ULL;
}

int shm_producer_claim(ShmHeader* hdr, uint64_t owner) {
    uint64_t empty = 0;
    return owner != 0 && __atomic_compare_exchange_n(&hdr->producer_claim,
        &empty, owner, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

void shm_producer_release(ShmHeader* hdr, uint64_t owner) {
    if (owner) __atomic_compare_exchange_n(&hdr->producer_claim, &owner, 0,
                                         0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

int shm_producer_reserve(ShmProducerContext* ctx, uint64_t need, uint64_t* dropped_out) {
    uint64_t write = ctx->cached_write;
    *dropped_out = 0;
    if (!need || (need & 7) || need > ctx->capacity || write > UINT64_MAX - need)
        return 0;
    for (;;) {
        uint64_t read = __atomic_load_n(&ctx->hdr->read_pos, __ATOMIC_ACQUIRE);
        uint64_t end = read, dropped = 0;
        if (read > write || write - read > ctx->capacity || ((read | write) & 7))
            return 0;
        ctx->cached_read = read;
        if (ctx->capacity - (write - read) >= need) return 1;
        do {
            uint64_t len = shm_read_u64_le(ctx->data, end, ctx->mask);
            uint64_t span = shm_record_span(len);
            if (!span || span > ctx->capacity || span > write - end) return 0;
            end += span;
            dropped++;
        } while (ctx->capacity - (write - end) < need);
        if (__atomic_compare_exchange_n(&ctx->hdr->read_pos, &read, end, 0,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            __atomic_fetch_add(&ctx->hdr->dropped_count, dropped, __ATOMIC_RELAXED);
            ctx->cached_read = end;
            *dropped_out = dropped;
            return 1;
        }
        /* A failed CAS means the consumer retired at least one record. The
         * old drop count is discarded; the next attempt starts from its cursor. */
    }
}

void shm_producer_write(ShmProducerContext* ctx, const unsigned char* payload,
                        size_t len, uint64_t pos) {
    shm_write_u64_le(ctx->data, pos, ctx->mask, (uint64_t)len);
    shm_copy_payload_into_ring(ctx->data, pos + 8, ctx->mask, payload, len);
}

void shm_producer_publish(ShmProducerContext* ctx, uint64_t end) {
    __atomic_store_n(&ctx->hdr->write_pos, end, __ATOMIC_RELEASE);
    __atomic_store_n(&ctx->hdr->latest_insert_time_ns,
                     (uint64_t)c_time_monotonic_ns(), __ATOMIC_RELEASE);
    ctx->cached_write = end;
}

int shm_producer_insert(ShmProducerContext* ctx, const unsigned char* payload,
                        size_t len, uint64_t* dropped_out) {
    uint64_t span = shm_record_span((uint64_t)len);
    if (!shm_producer_reserve(ctx, span, dropped_out)) return 0;
    shm_producer_write(ctx, payload, len, ctx->cached_write);
    shm_producer_publish(ctx, ctx->cached_write + span);
    return 1;
}

int shm_snapshot_begin(ShmConsumerContext* ctx, uint64_t* read_out, uint64_t* write_out) {
    uint64_t read = __atomic_load_n(&ctx->hdr->read_pos, __ATOMIC_ACQUIRE);
    uint64_t write = __atomic_load_n(&ctx->hdr->write_pos, __ATOMIC_ACQUIRE);
    if (read > write || write - read > ctx->capacity || ((read | write) & 7)) return 0;
    *read_out = read;
    *write_out = write;
    return 1;
}

int shm_snapshot_record(ShmConsumerContext* ctx, uint64_t pos, uint64_t write,
                        uint64_t* len_out) {
    uint64_t len, span;
    if ((pos & 7) || pos > write || write - pos < 8 || write - pos > ctx->capacity)
        return 0;
    len = shm_read_u64_le(ctx->data, pos, ctx->mask);
    span = shm_record_span(len);
    if (!span || span > ctx->capacity || span > write - pos) return 0;
    *len_out = len;
    return 1;
}

int shm_snapshot_valid(ShmConsumerContext* ctx, uint64_t read) {
    /* Any relaxed speculative word load that observes an overwriting release
     * store synchronizes with this acquire fence. The producer's preceding
     * retirement therefore happens before the following cursor load/CAS:
     * atomic coherence prevents validating/retiring the obsolete read cursor.
     * If no overwrite was observed, publication acquired in snapshot_begin
     * orders every original word. No ordinary access races on shared storage. */
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return __atomic_load_n(&ctx->hdr->read_pos, __ATOMIC_ACQUIRE) == read;
}

uint64_t shm_message_count(ShmConsumerContext* ctx) {
    for (;;) {
        uint64_t read, write, pos, len, count = 0;
        int valid = 1;
        if (!shm_snapshot_begin(ctx, &read, &write)) return 0;
        for (pos = read; pos < write;) {
            if (!shm_snapshot_record(ctx, pos, write, &len)) { valid = 0; break; }
            pos += shm_record_span(len);
            count++;
        }
        if (shm_snapshot_valid(ctx, read)) return valid ? count : 0;
    }
}

int shm_consumer_peek_available(ShmConsumerContext* ctx, uint64_t* len_out,
                                uint64_t* read_out) {
    uint64_t read, write, len;
    if (!shm_snapshot_begin(ctx, &read, &write) ||
        !shm_snapshot_record(ctx, read, write, &len) ||
        !shm_snapshot_valid(ctx, read)) return 0;
    *len_out = len;
    *read_out = read;
    return 1;
}

int shm_consumer_consume(ShmConsumerContext* ctx, unsigned char* dst,
                         uint64_t len, uint64_t read) {
    uint64_t current, write, actual, expected = read;
    if (!shm_snapshot_begin(ctx, &current, &write) || current != read ||
        !shm_snapshot_record(ctx, read, write, &actual) || actual != len) return 0;
    shm_copy_from_ring(dst, ctx->data, read + 8, ctx->mask, (size_t)len, ctx->capacity);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (!__atomic_compare_exchange_n(&ctx->hdr->read_pos, &expected,
        read + shm_record_span(len), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    __atomic_store_n(&ctx->hdr->latest_consume_time_ns,
                     (uint64_t)c_time_monotonic_ns(), __ATOMIC_RELEASE);
    return 1;
}

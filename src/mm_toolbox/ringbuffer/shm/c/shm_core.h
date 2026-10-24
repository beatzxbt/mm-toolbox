#ifndef SHM_CORE_H
#define SHM_CORE_H
#include "shm_types.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    ShmHeader* hdr;
    unsigned char* data;
    uint64_t capacity;
    uint64_t mask;
    uint64_t cached_read;
    uint64_t cached_write;
} ShmProducerContext;

typedef struct {
    ShmHeader* hdr;
    unsigned char* data;
    uint64_t capacity;
    uint64_t mask;
    int spin_wait;
} ShmConsumerContext;

/* One active writer per claimed ring; one active consuming operation. Closing
 * or releasing the mapping concurrently with an operation is unsupported.
 * Positions never wrap: insertion fails before uint64_t exhaustion. */
uint64_t shm_record_span(uint64_t payload_len);
int shm_producer_claim(ShmHeader* hdr, uint64_t owner);
void shm_producer_release(ShmHeader* hdr, uint64_t owner);
int shm_producer_reserve(ShmProducerContext* ctx, uint64_t need, uint64_t* dropped_out);
void shm_producer_write(ShmProducerContext* ctx, const unsigned char* payload,
                        size_t payload_len, uint64_t pos);
void shm_producer_publish(ShmProducerContext* ctx, uint64_t end);
int shm_producer_insert(ShmProducerContext* ctx, const unsigned char* payload,
                        size_t payload_len, uint64_t* dropped_out);
int shm_snapshot_begin(ShmConsumerContext* ctx, uint64_t* read_out, uint64_t* write_out);
int shm_snapshot_record(ShmConsumerContext* ctx, uint64_t pos, uint64_t write,
                        uint64_t* len_out);
int shm_snapshot_valid(ShmConsumerContext* ctx, uint64_t read);
uint64_t shm_message_count(ShmConsumerContext* ctx);
int shm_consumer_peek_available(ShmConsumerContext* ctx, uint64_t* msg_len_out,
                                uint64_t* read_pos_out);
/* dst is private scratch: a failed retirement may have copied speculative
 * words into it. The caller must discard them and retry without exposing dst. */
int shm_consumer_consume(ShmConsumerContext* ctx, unsigned char* dst,
                         uint64_t msg_len, uint64_t read_pos);
#endif

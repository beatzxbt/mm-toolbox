/* Standalone native regressions, also built with ASan/UBSan by release checks.
 * The compile-time word-load hook deterministically pauses a speculative copy
 * while the producer retires/overwrites that exact record. */
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "shm_core.h"
#include "shm_helpers.h"

static ShmProducerContext* paused_producer;
static int hook_countdown;
static uint64_t hook_dropped;

void shm_test_after_read_word(void) {
    if (hook_countdown > 0 && --hook_countdown == 0) {
        unsigned char replacement[40];
        memset(replacement, 'X', sizeof replacement);
        assert(shm_producer_insert(paused_producer, replacement,
                                    sizeof replacement, &hook_dropped));
    }
}

static void deterministic(void) {
    ShmHeader header = {0};
    uint64_t words[8] = {0};
    unsigned char* data = (unsigned char*)words;
    unsigned char out[64] = {0};
    ShmProducerContext producer = {&header, data, 64, 63, 0, 0};
    ShmConsumerContext consumer = {&header, data, 64, 63, 1};
    uint64_t dropped, length, read;
    for (int i = 0; i < 4; i++)
        assert(shm_producer_insert(&producer, (const unsigned char*)"AAAAAAAA", 8, &dropped));
    assert(shm_consumer_peek_available(&consumer, &length, &read));
    unsigned char replacement[40];
    memset(replacement, 'X', sizeof replacement);
    assert(shm_producer_insert(&producer, replacement, sizeof replacement, &dropped));
    assert(dropped == 3 && header.dropped_count == 3 && header.read_pos == 48);
    assert(!shm_consumer_consume(&consumer, out, length, read));
    assert(header.read_pos == 48 && shm_message_count(&consumer) == 2);
    assert(shm_consumer_peek_available(&consumer, &length, &read));
    assert(shm_consumer_consume(&consumer, out, length, read));
    assert(length == 8 && memcmp(out, "AAAAAAAA", 8) == 0);
    assert(shm_consumer_peek_available(&consumer, &length, &read));
    assert(shm_consumer_consume(&consumer, out, length, read));
    assert(length == 40 && memcmp(out, replacement, 40) == 0);
    assert(shm_message_count(&consumer) == 0 && header.read_pos == header.write_pos);

    /* Native copy actually sees one original and subsequent replacement word.
     * No stale/mixed scratch may be returned as a successful consumption. */
    memset(&header, 0, sizeof header);
    memset(words, 0, sizeof words);
    producer.cached_read = producer.cached_write = 0;
    unsigned char original[24];
    memset(original, 'A', sizeof original);
    assert(shm_producer_insert(&producer, original, sizeof original, &dropped));
    assert(shm_producer_insert(&producer, original, sizeof original, &dropped));
    assert(shm_consumer_peek_available(&consumer, &length, &read));
    paused_producer = &producer;
    hook_countdown = 2; /* header read, then first payload word */
    assert(!shm_consumer_consume(&consumer, out, length, read));
    assert(hook_countdown == 0 && hook_dropped == 2 && header.read_pos == 64);
    assert(shm_message_count(&consumer) == 1);
    assert(!shm_snapshot_valid(&consumer, read));
    assert(shm_consumer_peek_available(&consumer, &length, &read));
    assert(shm_consumer_consume(&consumer, out, length, read));
    assert(length == 40 && memcmp(out, replacement, 40) == 0);

    /* Consumer retirement first: a later producer reuses it without counting
     * it as a drop, and cannot invalidate the already-private result. */
    assert(shm_producer_insert(&producer, original, sizeof original, &dropped));
    assert(shm_consumer_peek_available(&consumer, &length, &read));
    assert(shm_consumer_consume(&consumer, out, length, read));
    assert(shm_producer_insert(&producer, replacement, sizeof replacement, &dropped));
    assert(dropped == 0 && length == 24 && memcmp(out, original, 24) == 0);

    /* Invalid headers and near-wrap positions never allocate/copy or regress. */
    shm_write_u64_le(data, header.read_pos, 63, UINT64_MAX);
    assert(!shm_consumer_peek_available(&consumer, &length, &read));
    assert(shm_message_count(&consumer) == 0);
    header.read_pos = header.write_pos = UINT64_MAX - 7;
    producer.cached_write = header.write_pos;
    assert(!shm_producer_insert(&producer, original, 1, &dropped));
    assert(header.read_pos == UINT64_MAX - 7 && header.write_pos == UINT64_MAX - 7);
    puts("deterministic interleavings, drop accounting and bounds passed");
}

#define STRESS_MESSAGES 200000
struct stress_state {
    ShmHeader header;
    uint64_t words[32];
    uint64_t consumed;
    int done;
};

static void* writer(void* value) {
    struct stress_state* state = value;
    ShmProducerContext producer = {&state->header, (unsigned char*)state->words, 256, 255, 0, 0};
    unsigned char payload[96];
    uint64_t dropped;
    for (uint64_t i = 1; i <= STRESS_MESSAGES; i++) {
        size_t length = 8 + i % 89;
        memcpy(payload, &i, 8);
        memset(payload + 8, (int)(i % 251), length - 8);
        assert(shm_producer_insert(&producer, payload, length, &dropped));
    }
    __atomic_store_n(&state->done, 1, __ATOMIC_RELEASE);
    return NULL;
}

static void* reader(void* value) {
    struct stress_state* state = value;
    ShmConsumerContext consumer = {&state->header, (unsigned char*)state->words, 256, 255, 1};
    uint64_t length, read, sequence, previous = 0, cursor = 0;
    unsigned char payload[256];
    for (;;) {
        uint64_t current = __atomic_load_n(&state->header.read_pos, __ATOMIC_ACQUIRE);
        assert(current >= cursor);
        cursor = current;
        assert(shm_message_count(&consumer) <= 32);
        if (shm_consumer_peek_available(&consumer, &length, &read) &&
            shm_consumer_consume(&consumer, payload, length, read)) {
            assert(length >= 8 && length <= 96);
            memcpy(&sequence, payload, 8);
            assert(sequence > previous && sequence <= STRESS_MESSAGES);
            assert(length == 8 + sequence % 89);
            for (size_t j = 8; j < length; j++) assert(payload[j] == sequence % 251);
            previous = sequence;
            state->consumed++;
            if (!(state->consumed % 19)) sched_yield();
        } else if (__atomic_load_n(&state->done, __ATOMIC_ACQUIRE) &&
            __atomic_load_n(&state->header.read_pos, __ATOMIC_ACQUIRE) ==
            __atomic_load_n(&state->header.write_pos, __ATOMIC_ACQUIRE)) break;
    }
    return NULL;
}

static void stress(void) {
    struct stress_state state = {0};
    pthread_t producer, consumer;
    assert(!pthread_create(&producer, NULL, writer, &state));
    assert(!pthread_create(&consumer, NULL, reader, &state));
    assert(!pthread_join(producer, NULL));
    assert(!pthread_join(consumer, NULL));
    assert(state.consumed + state.header.dropped_count == STRESS_MESSAGES);
    printf("concurrency passed: consumed=%llu dropped=%llu\n",
           (unsigned long long)state.consumed, (unsigned long long)state.header.dropped_count);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    assert(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void benchmark(void) {
    const size_t lengths[] = {8, 64, 256, 4096, 65528};
    uint64_t* words = calloc(8192, sizeof(uint64_t));
    unsigned char* payload = calloc(65536, 1);
    unsigned char* out = calloc(65536, 1);
    assert(words && payload && out);
    for (size_t item = 0; item < sizeof(lengths) / sizeof(lengths[0]); item++) {
        ShmHeader header = {0};
        ShmProducerContext producer = {&header, (unsigned char*)words, 65536, 65535, 0, 0};
        ShmConsumerContext consumer = {&header, (unsigned char*)words, 65536, 65535, 1};
        uint64_t dropped, length, read, start = monotonic_ns();
        const int iterations = 20000;
        for (int i = 0; i < iterations; i++) {
            assert(shm_producer_insert(&producer, payload, lengths[item], &dropped));
            assert(shm_consumer_peek_available(&consumer, &length, &read));
            assert(shm_consumer_consume(&consumer, out, length, read));
        }
        printf("payload=%zu roundtrip_ns=%.1f\n", lengths[item],
               (double)(monotonic_ns() - start) / iterations);
    }
    free(words);
    free(payload);
    free(out);
}

int main(int argc, char** argv) {
    if (argc > 1 && !strcmp(argv[1], "benchmark")) benchmark();
    else { deterministic(); stress(); }
    return 0;
}

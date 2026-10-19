#ifndef SHM_TYPES_H
#define SHM_TYPES_H

#include <stdint.h>

/* Format 2: aligned atomic words, padded records, exclusive writer leases.
 * Changing the magic rejects every mapping made by the previous byte layout. */
#define SHM_MSG_HEADER_SIZE 8
#define SHM_MAGIC 0x3253484252ULL
#define SHM_MPSC_MAGIC 0x3253484D50ULL

/* Each mapping and sub-ring starts on a 64-byte boundary. Keep these fields
 * uint64_t: all accesses while mapped are through lock-free 64-bit atomics. */
typedef struct {
    uint64_t magic;
    uint64_t capacity;
    uint64_t mask;
    uint64_t write_pos;
    uint64_t read_pos;
    uint64_t producer_claim;
    uint64_t latest_insert_time_ns;
    uint64_t latest_consume_time_ns;
    uint64_t dropped_count;
    uint64_t reserved[7];
} ShmHeader;

typedef struct {
    uint64_t magic;
    uint64_t num_rings;
    uint64_t ring_capacity;
    uint64_t next_producer_ring; /* rotating search start; claims require per-ring CAS */
    uint64_t reserved[4];
} ShmMpscGlobalHeader;

/* Identical type, rather than an incompatible struct cast. */
typedef ShmHeader ShmSubRingHeader;

#if defined(__GNUC__) || defined(__clang__)
_Static_assert(__atomic_always_lock_free(8, 0),
               "SHM requires native lock-free 64-bit atomics");
#else
#error "SHM requires compiler lock-free atomic builtins"
#endif
_Static_assert(sizeof(ShmHeader) == 128, "SHM format 2 header size");
_Static_assert(sizeof(ShmMpscGlobalHeader) == 64, "SHM global header size");

#endif

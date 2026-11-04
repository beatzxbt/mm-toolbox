# Shared memory queue format 2

SPSC queues and every MPSC shard have one exclusively claimed producer.
Creating an MPSC queue also claims a shard: provision `num_rings` for the
creator plus every simultaneously attached producer. Attachment raises
`RuntimeError` when all shards are claimed. `close()` and normal finalization
release a claim while preserving queued records. A process must attach a new
producer after `fork`; inherited objects cannot write or release the parent's
claim. Claims left by a crashed process are not stolen automatically. Recreate
the mapping after its users have stopped to recover that capacity.

Each record contains an eight-byte length followed by the payload padded to
an eight-byte boundary. All shared record accesses use aligned, native,
lock-free 64-bit atomic operations. A compile-time check rejects platforms
without those operations. Ring and sub-ring headers occupy 128 bytes; the
MPSC global header occupies 64 bytes. Format 2 magic values reject older
mappings. Capacities below eight bytes are rounded up to eight.

The producer makes space by CAS-advancing the retirement cursor past complete
oldest records. The consumer copies into private storage, then CAS-advances
that same cursor. Only a consumer that wins its CAS exposes the copy. No
producer waits for a consumer to finish, and no mutex, spinlock or reader busy
flag coordinates the native operations. Blocking wrappers wait for data when
empty. A consumer may retry under continuous overwrite.

Atomic payload stores use release ordering; speculative loads use relaxed
ordering. An acquire fence follows speculative reads before retirement or
snapshot validation. If any load observes replacement storage, its producer's
release store synchronizes with that fence. The producer's preceding cursor
retirement then happens before the consumer's cursor CAS/load. Atomic
coherence prevents a successful stale CAS or a valid stale snapshot. Original
records are ordered by acquire-loading the published write boundary. No
ordinary shared-memory copy is used on either side.

Positions are monotonic unsigned 64-bit byte offsets and never wrap. Inserts
fail before an offset addition would overflow; recreate the mapping after
stopping its users if this lifetime limit is reached. A producer crash after
retiring old records and before publishing a replacement can lose those
retired records, but cannot publish a partial replacement or block a reader.
Closing/unmapping concurrently with an active operation is unsupported.

`insert()` and `insert_batch()` retain their boolean success result. Inspect
`last_insert_dropped` immediately afterward for that operation's overwritten
record count. `dropped_count` is the shared cumulative count since mapping
creation (summed across MPSC shards), and `failed_inserts` counts rejected
operations for that producer instance. Only successful producer retirement
CAS operations increment drop counts. Timestamp properties remain monotonic
clock readings.

There is no separately incremented/decremented message count. `len()` scans
validated record headers within an acquired publication boundary and checks
the retirement cursor afterward. It cannot underflow and never establishes
whether a consuming operation succeeds. Counts and snapshots are coherent
within one shard; combining independently active shards is not a simultaneous
global snapshot. `consume_into()` and `consume_all_into()` use reusable private
scratch that grows only for a new largest payload. A failed speculative
retirement leaves caller storage untouched, with no per-message Python
allocation after the scratch has reached the required capacity.

Native regressions are in `tests/ringbuffer/shm_native_regression.c`. The test
hook forces overwrite between atomic payload word loads, and pthread stress
checks payload integrity, monotonic cursors and consumed-plus-dropped
accounting. Its `benchmark` mode reports native round-trip timings for 8, 64,
256, 4096 and 65528-byte payloads. Performance and weak-memory validation must
be checked on each supported architecture.

## Benchmark results (2026-11-04)

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `c5f9dba`.
These are local measurements and do not establish performance on other machines.

### Python SPSC

The p50 latency columns are nanoseconds per insert/consume from 100,000 samples
per operation. Each paired throughput run sent and consumed exactly 100,000
messages; the runner checks that the producer and consumer counts match.

| Payload (bytes) | Insert p50 (ns) | Consume p50 (ns) | Producer throughput (msg/s) | Consumer throughput (msg/s) |
| --- | ---: | ---: | ---: | ---: |
| 32 | 125 | 291 | 7,462,687 | 7,518,797 |
| 128 | 125 | 250 | 7,042,254 | 7,092,199 |
| 512 | 125 | 334 | 5,649,718 | 5,714,286 |
| 2,048 | 250 | 458 | 2,544,529 | 2,570,694 |
| 8,192 | 1,084 | 1,500 | 659,196 | 639,386 |

```sh
.venv/bin/python -u benchmarks/ringbuffer/benchmark_shm_spsc.py --operations 100000 --warmup 1000 --multi-size --capacity-bytes 1048576 --latency-iterations 100000 --messages 100000
```

### Python MPSC

Latency p50s are nanoseconds per insert/consume; sample counts were
capacity-limited to 2,730 at 128 B, 1,365 at 512 B and 819 at 1,024 B. Throughput
scenarios ran for one second each. Inserted counts are successful producer
inserts; every row satisfied `inserted = overwritten + consumed + remaining`,
and failed inserts were zero. Scalability and fairness rows used 128-byte
payloads.

| Scenario | Inserted | Overwritten | Consumed | Remaining |
| --- | ---: | ---: | ---: | ---: |
| 128 B, 1 producer | 4,653,383 | 0 | 4,653,383 | 0 |
| 128 B, 2 producers | 9,158,398 | 8,677 | 9,149,721 | 0 |
| 128 B, 4 producers | 16,276,896 | 11,104,617 | 5,171,855 | 424 |
| 512 B, 1 producer | 3,534,797 | 5,277 | 3,529,520 | 0 |
| 512 B, 2 producers | 7,903,915 | 2,584,238 | 5,319,677 | 0 |
| 512 B, 4 producers | 13,669,315 | 10,406,037 | 3,263,278 | 0 |
| 1,024 B, 1 producer | 3,317,235 | 3,382 | 3,313,853 | 0 |
| 1,024 B, 2 producers | 6,838,497 | 4,006,110 | 2,832,387 | 0 |
| 1,024 B, 4 producers | 12,746,241 | 10,040,456 | 2,705,785 | 0 |
| Scalability, 128 B, 1 producer | 4,342,402 | 395 | 4,342,007 | 0 |
| Scalability, 128 B, 4 producers | 16,677,744 | 11,845,888 | 4,831,676 | 180 |
| Fairness, 128 B, 4 producers | 17,587,654 | 13,806,061 | 3,779,331 | 2,262 |

| Payload (bytes) | Insert p50 (ns) | Consume p50 (ns) | Samples per operation |
| --- | ---: | ---: | ---: |
| 128 | 125 | 250 | 2,730 |
| 512 | 125 | 375 | 1,365 |
| 1,024 | 167 | 417 | 819 |

```sh
.venv/bin/python -u benchmarks/ringbuffer/benchmark_shm_mpsc.py --operations 100000 --warmup 1000 --multi-size --capacity-bytes 1048576 --num-rings 0 --latency-iterations 4096 --duration 1.0 --throughput-producer-counts 1,2,4 --scalability-producer-counts 1,4
```

### Native core round trip

The existing C regression tool measures the mean native producer-insert plus
consumer-consume round trip over 20,000 iterations per payload. It is a
single-process microbenchmark of the core, not a file-backed Python latency
measurement.

| Payload (bytes) | Mean round trip (ns) |
| --- | ---: |
| 8 | 154.3 |
| 64 | 172.7 |
| 256 | 216.6 |
| 4,096 | 1,136.5 |
| 65,528 | 8,108.9 |

```sh
cc -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -pthread -Isrc/mm_toolbox/ringbuffer/shm/c tests/ringbuffer/shm_native_regression.c src/mm_toolbox/ringbuffer/shm/c/shm_core.c src/mm_toolbox/ringbuffer/shm/c/shm_helpers.c src/mm_toolbox/time/ctime_impl.c -o /tmp/mm-toolbox-shm-native-regression-benchmark
/tmp/mm-toolbox-shm-native-regression-benchmark benchmark
```

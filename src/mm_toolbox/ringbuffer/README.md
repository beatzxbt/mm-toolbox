# Ring buffers

High-performance ring buffers for in-process queues and shared-memory transport.
The module includes byte, numeric, generic, and shared-memory variants with
consistent insert/consume semantics.

## Architecture overview

### In-process circular buffer

All in-process buffers use a power-of-two circular array that overwrites the
oldest entry on overflow. Writes advance `head`, reads advance `tail`.

```
            head (write)
               ▼
┌─────────────────────────┐
│  [ ] [ ] [ ] [ ] [ ]    │  <- fixed capacity (power of two)
└─────────────────────────┘
   ▲
 tail (oldest)

insert()    -> writes at head, advances head
consume()   -> pops oldest (tail), FIFO
unwrapped() -> oldest to newest
```

### BytesRingBufferFast slot layout

`BytesRingBufferFast` pre-allocates fixed-size slots to reduce allocations and
copy overhead. Oversized messages raise immediately.

```
slots (fixed size)
┌────────┬────────┬────────┬────────┐
│ slot 0 │ slot 1 │ slot 2 │ slot 3 │
└────────┴────────┴────────┴────────┘
   ▲                    ▲
  tail                head

insert() -> memcpy into slot[head], advance head
```

### Shared memory (SPSC)

The shared-memory ring is a file-backed mmap with length-prefixed messages and
single-producer/single-consumer semantics.

```
Producer                       Consumer
┌──────────────┐               ┌──────────────┐
│ write len+msg│               │ read len+msg │
└──────┬───────┘               └──────┬───────┘
       │                              ▲
       ▼                              │
┌──────────────────────────────┐     │
│ mmap file (power-of-two ring)│─────┘
└──────────────────────────────┘
```

### Shared memory (MPSC)

`ShmMpscProducer` and `ShmMpscConsumer` provide a multi-producer/single-consumer
shared-memory ring buffer built on a sharded sub-ring architecture. Each
sub-ring is a complete SPSC queue; producers claim an exclusive sub-ring using CAS, starting from a rotating
search position while the consumer round-robins across all sub-rings for fairness.

- File-backed mmap with power-of-two capacity per sub-ring.
- Length-prefixed messages.
- Blocking consumer loops with spin/yield semantics.
- `insert_batch` on the producer batches multiple items into a single sub-ring
  with one commit.

## In-process buffers

### BytesRingBuffer
- Stores `bytes` entries in a Python list.
- Optional `only_insert_unique` avoids duplicate inserts (linear scan).
- Async methods use an `asyncio.Event` (disable with `disable_async=True`).
- Consumes in FIFO order (`consume()` returns the oldest queued item).
- Best for moderate throughput with flexible message sizes.

### BytesRingBufferFast
- Pre-allocated slots sized from `expected_item_size` and `buffer_percent`.
- `insert_char` is Cython-only for direct `char*` ingestion; Python uses `insert(bytes)`.
- Oversized inserts raise `ValueError`; no truncation occurs.
- Best for high-frequency byte workloads with predictable message sizes.

### GenericRingBuffer
- Stores arbitrary Python objects with minimal overhead.
- Same API as the bytes buffer, minus deduplication.
- Supports constant-time integer indexing in FIFO order without consuming or
  copying contents; negative indices count back from the newest item.
- Best for in-process queues of Python objects.

### NumericRingBuffer
- Stores numeric types in a NumPy-backed buffer.
- Accepts `int`/`uint` (1/2/4/8 bytes) and `float` (4/8 bytes).
- Best for numeric streams where NumPy interop and slicing are important.

## Shared memory buffers

### SPSC (`ShmSpscProducer` / `ShmSpscConsumer`)
- File-backed mmap ring buffer for bytes.
- Single-producer/single-consumer only.
- Length-prefixed messages with power-of-two capacity.
- Blocking consumer loops with spin/yield semantics.

### MPSC (`ShmMpscProducer` / `ShmMpscConsumer`)
- File-backed mmap ring buffer for bytes.
- Multi-producer/single-consumer via sharded sub-rings.
- Length-prefixed messages with power-of-two capacity per sub-ring.
- Blocking consumer loops with spin/yield semantics.
- Producer supports `insert`, `insert_char`, and `insert_batch`.

## Quick start

### BytesRingBuffer

```python
from mm_toolbox.ringbuffer import BytesRingBuffer

rb = BytesRingBuffer(max_capacity=128)
rb.insert(b"alpha")
rb.insert(b"beta")

oldest = rb.consume()        # b"alpha"
all_items = rb.consume_all() # [b"beta"]
```

### NumericRingBuffer

```python
from mm_toolbox.ringbuffer import NumericRingBuffer

rb = NumericRingBuffer(max_capacity=8, dtype=np.float64)
rb.insert(1.0)
rb.insert(2.0)
values = rb.unwrapped()
```

## Protocols

The module exports protocol types for type-checked abstractions:

- `RingBufferProtocol` – common interface for all ring buffers.
- `AsyncRingBufferProtocol` – async-capable ring buffer interface.
- `RingBufferConsumerProtocol` – consumer-side operations.
- `RingBufferProducerProtocol` – producer-side operations.
- `SupportsAsyncConsume` – trait for async consumption.

## Behavior notes

- Capacity rounds up to the next power of two for fast masking.
- When full, inserts overwrite the oldest element.
- In-process buffers use FIFO consume semantics: `consume()` pops the oldest item.
- `unwrapped()` returns items from oldest to newest.
- Async helpers (`aconsume`, `aconsume_iterable`) block until new data arrives.

## Shared-memory format and loss reporting

The hardened format uses eight-byte-aligned, padded records and a versioned
128-byte per-ring header. Existing mappings from previous versions are rejected;
stop all users and recreate the mapping when upgrading. Capacity includes record
headers and up to seven padding bytes per message.

The native queue core uses lock-free 64-bit atomic copies and cursor CAS; it has
no reader/writer lock. A failed speculative read is retried without delivering
its contents. Blocking wrappers still wait for data when empty.

Each producer instance owns one shard until close. More active producers than
shards raises `RuntimeError`; a creator that also writes counts as a producer.
Do not use an inherited producer after fork: attach a fresh instance. Claims
from crashed writers are not stolen; recreate the mapping after stopping users.
One consuming operation per SPSC ring or MPSC consumer is supported. Closing or
unmapping concurrently with an operation is unsupported.

`dropped_count`, `last_insert_dropped`, and `failed_inserts` expose overwrite and
insertion failures. Callers requiring stream continuity must handle losses.
`consume_into` reuses private scratch after its maximum payload capacity has been
established; a failed retirement does not modify the destination. `len` and
snapshot inspection validate their copied record view and can retry under load.

## Benchmark results (2026-11-04)

The p50 values are nanoseconds per call. Numeric used a 1,024-slot `float32`
buffer; bytes used a 1,024-slot buffer with async and unique filtering disabled.
Each operation had 100,000 measured samples after 1,000 warmups, except batch
operations, whose sample counts scale with the batch size.

### Numeric ring buffer

| Operation | p50 (ns/call) |
| --- | ---: |
| insert | 291 |
| consume | 1,334 |
| contains | 500 |
| peekleft | 83 |
| peekright | 83 |
| unwrapped | 250 |
| fill+consume_all | 28,333 |
| insert_batch, n=1 | 334 |
| insert_batch, n=10 | 625 |
| insert_batch, n=100 | 3,084 |
| insert_batch, n=1,000 | 28,000 |

```sh
.venv/bin/python -u benchmarks/ringbuffer/benchmark_numeric.py --operations 100000 --warmup 1000 --size 1024
```

### Bytes ring buffer

| Payload (bytes) | Insert | Contains | Consume | Fill + consume all |
| --- | ---: | ---: | ---: | ---: |
| 32 | 208 | 292 | 291 | 6,334 |
| 128 | 167 | 292 | 291 | 7,250 |
| 512 | 209 | 292 | 375 | 7,791 |
| 2,048 | 250 | 458 | 375 | 15,417 |

Values are p50 ns/call. `peekleft` and `peekright` were 42 ns/call at p50 for
all four payload sizes. Batch insert p50s (ns/call), for batch sizes 1, 10, 100
and 1,000 respectively, were 32 B: 291/375/834/6,000; 128 B:
291/417/1,167/8,979.5; 512 B: 375/500/1,583/11,917; and 2,048 B:
375/500/1,750/15,750.

```sh
.venv/bin/python -u benchmarks/ringbuffer/benchmark_bytes.py --operations 100000 --warmup 1000 --size 1024
```

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `c5f9dba`.
These are local measurements and do not establish performance on other machines. See the
[shared-memory benchmark results](shm/README.md#benchmark-results-2026-11-04)
for SPSC, MPSC and native-core measurements.

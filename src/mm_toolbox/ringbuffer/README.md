# Ring buffers

High-performance ring buffers for in-process queues and inter-process transport.
The module includes byte, numeric, generic, IPC, and shared-memory variants with
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

### IPC transport (ZMQ)

IPC buffers use PUSH/PULL sockets with a configurable backlog and simple
topologies (SPSC/MPSC/SPMC, but not MPMC).

```
Producers                         Consumer
┌──────────────┐                  ┌──────────────┐
│ insert(bytes)│                  │ consume()    │
└──────┬───────┘                  └──────┬───────┘
       │                                ▲
       ▼                                │
┌───────────────────────────┐          │
│ ZMQ PUSH/PULL channel     │──────────┘
└───────────────────────────┘
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

## In-process buffers

### BytesRingBuffer
- Stores `bytes` entries in a Python list.
- Optional `only_insert_unique` avoids duplicate inserts (linear scan).
- Async methods use an `asyncio.Event` (disable with `disable_async=True`).
- Consumes in FIFO order (`consume()` returns the oldest queued item).
- Best for moderate throughput with flexible message sizes.

### BytesRingBufferFast
- Pre-allocated slots sized from `expected_item_size` and `buffer_percent`.
- `insert_char` supports direct `char*` ingestion to avoid conversions.
- Oversized inserts raise `ValueError`; no truncation occurs.
- Best for high-frequency byte workloads with predictable message sizes.

### GenericRingBuffer
- Stores arbitrary Python objects with minimal overhead.
- Same API as the bytes buffer, minus deduplication.
- Best for in-process queues of Python objects.

### NumericRingBuffer
- Stores numeric types in a NumPy-backed buffer.
- Accepts `int`/`uint` (1/2/4/8 bytes) and `float` (4/8 bytes).
- Best for numeric streams where NumPy interop and slicing are important.

## IPC buffer (ZMQ)

### IPCRingBufferProducer / IPCRingBufferConsumer
- PUSH/PULL sockets with high-water-mark backlog.
- Supports SPSC, MPSC, or SPMC (not MPMC).
- `insert_packed` / `consume_packed` batch multiple messages into one frame.
- `should_producer_bind()` decides bind/connect to avoid multiple binders.

## Shared memory buffer (SPSC)

### shm
- File-backed mmap ring buffer for bytes.
- Single-producer/single-consumer only.
- Length-prefixed messages with power-of-two capacity.
- Blocking consumer loops with spin/yield semantics.

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

rb = NumericRingBuffer(max_capacity=8, dtype=float)
rb.insert(1.0)
rb.insert(2.0)
values = rb.unwrapped()
```

### IPC ring buffer

```python
from mm_toolbox.ringbuffer import IPCRingBufferConfig, IPCRingBufferProducer

config = IPCRingBufferConfig.default()
producer = IPCRingBufferProducer(config)
producer.insert(b"payload")
producer.stop()
```

## Behavior notes

- Capacity rounds up to the next power of two for fast masking.
- When full, inserts overwrite the oldest element.
- In-process buffers use FIFO consume semantics: `consume()` pops the oldest item.
- `unwrapped()` returns items from oldest to newest.
- Async helpers (`aconsume`, `aconsume_iterable`) block until new data arrives.
- IPC buffers are transport-backed (ZMQ), not shared memory; use `shm` for SPSC.

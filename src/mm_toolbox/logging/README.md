# Logging

This package provides two logging stacks:
- Standard logger: single-process, synchronous buffering with optional remote-delivery workers.
- Advanced logger: multi-process logging with IPC transport and a master-worker
  architecture.

## Standard logger (single-process)

Use this when everything runs in one process and you want simple, buffered logging
with optional stdout mirroring.

Highlights:
- Flushes synchronously when the size threshold is met or a later log call observes that the configured interval elapsed.
- `flush()` and `shutdown()` explicitly flush pending messages, including when the logger has been quiet.
- Discord and Telegram delivery runs on bounded handler-owned workers, so an active asyncio loop is supported and slow endpoints do not block the logging call.
- Handler fan-out (file, Discord, Telegram).
- Buffered writes reduce handler overhead for high-frequency logs.

### Architecture overview

The standard logger keeps its message buffer in the caller's thread. It pushes
each flushed batch to local handlers synchronously. Discord and Telegram
handlers enqueue bounded delivery work on their own background threads.

```
┌──────────────┐
│  App Thread  │
│ logger.info  │
└──────┬───────┘
       │ append
       ▼
┌──────────────┐
│ Logger Buffer│
└──────┬───────┘
       │ flush condition or explicit flush()
       ▼
┌──────────────┐
│   Handlers   │
└──────┬───────┘
       │ remote handlers enqueue
       ▼
┌──────────────────────┐
│ Bounded delivery     │
│ workers (Discord,    │
│ Telegram)            │
└──────────────────────┘
```

Quick start:
```python
from mm_toolbox.logging.standard import Logger, LoggerConfig, LogLevel
from mm_toolbox.logging.standard import FileLogHandler

config = LoggerConfig(
    base_level=LogLevel.INFO,
    flush_on_interval=True,
    flush_interval_s=0.5,
    do_stdout=True,
)
logger = Logger(name="app", config=config, handlers=[FileLogHandler("app.log")])

logger.info("booted")
logger.error("something went wrong")
logger.shutdown()
```

`flush_on_interval=True` does not start a timer. The next log call checks the
elapsed time and flushes if due. Use `logger.flush()` when a quiet logger must
flush at a specific time; `shutdown()` also flushes before closing handlers.

When to choose:
- Best for single-process apps, notebooks, or services without forked workers.
- Not designed for cross-process aggregation (use the advanced logger for that).

## Advanced logger (multi-process)

High-performance, distributed logging for multi-process applications where
performance and reliability are critical.

### Architecture overview

The system uses a master-worker pattern with ShmMpscProducer/ShmMpscConsumer-based IPC:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ WorkerLogger│    │ WorkerLogger│    │ WorkerLogger│
│  (Process A)│    │  (Process B)│    │  (Process N)│
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                   │                   │
       │ .insert()         │ .insert()         │ .insert()
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
               ┌───────────▼───────────┐
               │    IPC Transport      │
               │ (MPSC: Multi→Single)  │
               └───────────┬───────────┘
                           │ .consume_all()
                           ▼
                 ┌─────────────────┐
                 │  MasterLogger   │
                 │  (Single Proc)  │
                 └─────────┬───────┘
                           │
                 ┌─────────▼───────┐
                 │    Handlers     │
                 │ File │Discord│  │
                 │ Telegram│etc.   │
                 └─────────────────┘
```

### Core components

#### WorkerLogger
- Purpose: lightweight logger for worker processes
- Batching: collects logs in memory, sends in batches for efficiency
- Heartbeats: sends periodic health checks (60s intervals)
- Non-blocking: never blocks the main process thread

#### MasterLogger
- Purpose: central aggregator that receives all logs
- Processing: decodes, filters, and routes logs to handlers
- Monitoring: tracks worker health via heartbeat monitoring
- Handlers: forwards logs to files, Discord, Telegram, etc.

#### Message types
- Log batches: binary-serialized log entries with timestamps
- Heartbeats: worker health signals with next check-in time
- Shutdown: special heartbeats indicating worker termination

### Performance features

#### Binary serialization
```cython
# Efficient binary format using memcpy
cdef bytes buffer = bytes(sizeof(u64) + sizeof(u8) + sizeof(u32) + msg_len)
memcpy(buffer, &time, sizeof(u64))        # Timestamp
memcpy(buffer + 8, &level, sizeof(u8))    # Log level
memcpy(buffer + 9, &msg_len, sizeof(u32)) # Message length
```

#### Batched operations
- Workers: collect logs locally, send batches every 1s (configurable)
- Master: processes all available messages in single `consume_all()` call
- Memory: dynamic buffer growth prevents pre-allocation waste

#### Zero-copy messaging
- Uses the project's `ShmMpscProducer`/`ShmMpscConsumer` transport for reliability and throughput
- MPSC pattern: multiple producers to single consumer
- Bounded backlog with backpressure semantics

### Configuration

```python
from mm_toolbox.logging.advanced import LoggerConfig, MasterLogger, WorkerLogger
from mm_toolbox.logging.advanced import FileLogHandler, DiscordLogHandler

config = LoggerConfig(
    base_level=LogLevel.INFO,
    path="/tmp/my_app_logger.shm",
    flush_interval_s=1.0,
    str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

handlers = [
    FileLogHandler("/var/log/myapp.log", create=True),
    DiscordLogHandler("https://discord.com/api/webhooks/..."),
]
master = MasterLogger(config, handlers)

worker = WorkerLogger(config, name="DataProcessor")
worker.info("Processing batch 123")
worker.error("Failed to process item", msg_bytes=error_data)
```

Additional config fields:
- `emit_internal` — emit internal startup/shutdown logs (default `False`).
- `max_batch_messages` — max messages before forced flush (default `10000`).
- `max_batch_bytes` — max bytes before forced flush (default `1MB`).
- `shm_capacity_bytes` — shared memory ring buffer capacity in bytes (default `64MB`).

The advanced logger derives its producer shard count from `shm_capacity_bytes`
and `max_batch_bytes`, with a maximum of 64 active producers including the
master. The defaults allocate 32 shards, leaving room for 31 worker loggers.
Each worker owns one shard; extra concurrent producers fail explicitly. The
worker clamps batches to a safe shard size and reports records that cannot fit.

### Message flow

#### 1. Worker logging
```python
worker.info("User login successful")
# Serialized to binary format
# Added to local batch buffer
# Sent via IPC when buffer fills or timeout hits
```

#### 2. Master processing
```python
messages = ipc_consumer.consume_all()
for msg in messages:
    if msg_type == LOG_BATCH:
        for handler in handlers:
            handler.push(decoded_logs)
    elif msg_type == HEARTBEAT:
        update_worker_health(worker_name)
```

#### 3. Health monitoring
```python
if current_time > next_checkin + 1_second:
    master.warning(f"Worker {name} appears dead")
```

### Error handling

- Connection failures: workers continue logging locally and reconnect
- Handler failures: individual handler errors do not affect others
- Worker crashes: master detects via missed heartbeats
- Graceful shutdown: flushes pending logs before exit

### Thread safety

- Shared-memory IPC: exclusively claimed producer shards use the lock-free queue core; worker-local batch operations use a thread lock.
- Process isolation: one worker crash does not publish a partial record. Its producer claim remains reserved until the mapping is recreated after all users stop.
- Atomic publication: complete binary records become visible after serialization and publication.

### When to use

Good for:
- Multi-process applications (trading systems, data pipelines)
- High-frequency logging (1000+ logs/sec)
- Remote log aggregation (Discord/Telegram notifications)
- Systems requiring high reliability

Overkill for:
- Single-process applications
- Low-frequency logging
- Simple file-only logging
- Development/debugging (use standard logging)

### Quick start

```python
master = MasterLogger(config, [FileLogHandler("app.log")])
worker = WorkerLogger(config, name=f"Worker-{os.getpid()}")

worker.info(b"Worker started")
worker.shutdown()
master.shutdown()
```

### Delivery and shutdown limits

Remote HTTP handlers check response status, retry transient failures at most
three times within a ten-second delivery budget, and honor Retry-After. If the
required delay exceeds the remaining budget, delivery fails explicitly without
an early retry. Background delivery errors go to the configured error callback
or stderr. Standard remote queues hold at most 256 deliveries; advanced handlers
allow at most 256 scheduled operations. Saturation is reported as loss.

Standard handler close and advanced handler/master shutdown use one bounded
drain deadline (two seconds by default). Pending work may be cancelled or
discarded at that deadline and is reported. A user-supplied synchronous handler
cannot be forcibly interrupted: a timed-out master leaves its mappings owned
by the drain thread until it can finish safely. Stop workers before the master
to include their final batches. File handlers append existing logs on restart.

`WorkerLogger.get_transport_stats()` returns cumulative `rejected_messages`,
`transport_errors` and `overwritten_batches`, including the final flush and
transport cleanup. Read it after `shutdown()` to inspect the final totals.
Overwritten batches can contain multiple log records; that count is not a
message count. Overwrite diagnostics are sampled while the counters retain
exact totals. These transport statistics do not acknowledge downstream handler
delivery. The advanced logging benchmark checks both final worker status and
observed handler consumption, and exits nonzero on lifecycle errors or missing
deliveries without recorded overwrites.

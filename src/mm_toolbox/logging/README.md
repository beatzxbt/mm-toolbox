# Logging

This package provides two logging stacks:
- Standard logger: single-process, low-overhead buffering with async handlers.
- Advanced logger: multi-process logging with IPC transport and a master-worker
  architecture.

## Standard logger (single-process)

Use this when everything runs in one process and you want simple, buffered logging
with optional stdout mirroring.

Highlights:
- Background thread flushes buffers on a cadence.
- Async handler fan-out (file, Discord, Telegram).
- Buffered writes reduce handler overhead for high-frequency logs.

### Architecture overview

The standard logger runs a background thread that drains a queue, batches
messages, and pushes them to handlers.

```
┌──────────────┐
│  App Thread  │
│ logger.info  │
└──────┬───────┘
       │ enqueue (log_msg, level)
       ▼
┌──────────────┐
│  Msg Queue   │
└──────┬───────┘
       │ drain + batch
       ▼
┌──────────────┐
│ Flush Thread │
│  (event loop)│
└──────┬───────┘
       │ async push
       ▼
┌──────────────────────┐
│ Handlers             │
│ File / Discord / ... │
└──────────────────────┘
```

Quick start:
```python
from mm_toolbox.logging.standard import Logger, LoggerConfig, LogLevel
from mm_toolbox.logging.standard import FileLogHandler

config = LoggerConfig(
    base_level=LogLevel.INFO,
    flush_interval_s=0.5,
    do_stdout=True,
)
logger = Logger(name="app", config=config, handlers=[FileLogHandler("app.log")])

logger.info("booted")
logger.error("something went wrong")
logger.shutdown()
```

When to choose:
- Best for single-process apps, notebooks, or services without forked workers.
- Not designed for cross-process aggregation (use the advanced logger for that).

## Advanced logger (multi-process)

High-performance, distributed logging for multi-process applications where
performance and reliability are critical.

### Architecture overview

The system uses a master-worker pattern with IPCRingBuffer-based IPC:

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
- Uses the project's `IPCRingBuffer` transport for reliability and throughput
- MPSC pattern: multiple producers to single consumer
- Bounded backlog with backpressure semantics

### Configuration

```python
from mm_toolbox.logging.advanced import LoggerConfig, MasterLogger, WorkerLogger
from mm_toolbox.logging.advanced import FileLogHandler, DiscordLogHandler

config = LoggerConfig(
    base_level=LogLevel.INFO,
    path="ipc:///tmp/my_app_logger",
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

- Lock-free design: uses message passing instead of shared memory
- Process isolation: worker crashes do not affect master or other workers
- Atomic operations: binary serialization with single-write semantics

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

worker.info("Worker started")
worker.shutdown()
master.shutdown()
```

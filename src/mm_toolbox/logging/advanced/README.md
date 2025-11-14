# Advanced Multi-Process Logging System

A high-performance, distributed logging system designed for multi-process applications where performance and reliability are critical.

## Architecture Overview

The system uses a **master-worker pattern** with IPCRingBuffer-based IPC for communication:

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

## Core Components

### WorkerLogger
- **Purpose**: Lightweight logger for worker processes
- **Batching**: Collects logs in memory, sends in batches for efficiency
- **Heartbeats**: Sends periodic health checks (60s intervals)
- **Non-blocking**: Never blocks the main process thread

### MasterLogger  
- **Purpose**: Central aggregator that receives all logs
- **Processing**: Decodes, filters, and routes logs to handlers
- **Monitoring**: Tracks worker health via heartbeat monitoring
- **Handlers**: Forwards logs to files, Discord, Telegram, etc.

### Message Types
- **Log Batches**: Binary-serialized log entries with timestamps
- **Heartbeats**: Worker health signals with next check-in time
- **Shutdown**: Special heartbeats indicating worker termination

## Performance Features

### Binary Serialization
```cython
# Efficient binary format using memcpy
cdef bytes buffer = bytes(sizeof(u64) + sizeof(u8) + sizeof(u32) + msg_len)
memcpy(buffer, &time, sizeof(u64))      # Timestamp
memcpy(buffer + 8, &level, sizeof(u8))  # Log level  
memcpy(buffer + 9, &msg_len, sizeof(u32)) # Message length
```

### Batched Operations
- **Workers**: Collect logs locally, send batches every 1s (configurable)
- **Master**: Processes all available messages in single `consume_all()` call
- **Memory**: Dynamic buffer growth prevents pre-allocation waste

### Zero-Copy Messaging
- Uses the project's `IPCRingBuffer` transport for reliability and throughput
- MPSC pattern: Multiple producers → Single consumer
- Bounded backlog with backpressure semantics

## Configuration

```python
from mm_toolbox.logging.advanced import LoggerConfig, MasterLogger, WorkerLogger
from mm_toolbox.logging.advanced import FileLogHandler, DiscordLogHandler

# Configure the logging system
config = LoggerConfig(
    base_level=LogLevel.INFO,
    path="ipc:///tmp/my_app_logger",  # IPC socket path
    flush_interval_s=1.0,             # Batch frequency
    str_format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Master process: aggregates and outputs logs
handlers = [
    FileLogHandler("/var/log/myapp.log", create=True),
    DiscordLogHandler("https://discord.com/api/webhooks/..."),
]
master = MasterLogger(config, handlers)

# Worker processes: generate logs
worker = WorkerLogger(config, name="DataProcessor")
worker.info("Processing batch 123")
worker.error("Failed to process item", msg_bytes=error_data)
```

## Message Flow

### 1. Worker Logging
```python
worker.info("User login successful")
# → Serialized to binary format
# → Added to local batch buffer
# → Sent via IPC when buffer fills or 1s timeout
```

### 2. Master Processing
```python
# Master's background thread:
messages = ipc_consumer.consume_all()  # Get all pending messages
for msg in messages:
    if msg_type == LOG_BATCH:
        for handler in handlers:
            handler.push(decoded_logs)     # → File, Discord, etc.
    elif msg_type == HEARTBEAT:
        update_worker_health(worker_name)
```

### 3. Health Monitoring
```python
# Workers send heartbeats every 60s
heartbeat = {
    "worker": "DataProcessor", 
    "current_time": 1234567890,
    "next_checkin": 1234567950  # +60s
}

# Master warns if worker misses check-in + 1s grace period
if current_time > next_checkin + 1_second:
    master.warning(f"Worker {name} appears dead")
```

## Error Handling

- **Connection failures**: Workers continue logging locally, reconnect automatically
- **Handler failures**: Individual handler errors don't affect others
- **Worker crashes**: Master detects via missed heartbeats
- **Graceful shutdown**: Flushes all pending logs before exit

## Thread Safety

- **Lock-free design**: Uses message passing instead of shared memory
- **Process isolation**: Worker crashes don't affect master or other workers
- **Atomic operations**: Binary serialization with single-write semantics

## When to Use

✅ **Good for:**
- Multi-process applications (trading systems, data pipelines)
- High-frequency logging (>1000 logs/sec)
- Remote log aggregation (Discord, Slack notifications)
- Systems requiring high reliability

❌ **Overkill for:**
- Single-process applications
- Low-frequency logging
- Simple file-only logging
- Development/debugging (use standard logging)

## Quick Start

```python
# In your main process
master = MasterLogger(config, [FileLogHandler("app.log")])

# In worker processes  
worker = WorkerLogger(config, name=f"Worker-{os.getpid()}")
worker.info("Worker started")

# Clean shutdown
worker.shutdown()  # Flushes remaining logs
master.shutdown() # Stops accepting new logs
```

The system handles the complexity of multi-process coordination while providing a simple logging interface similar to Python's standard library.

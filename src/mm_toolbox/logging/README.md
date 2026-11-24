# Logging

The standard logger buffers formatted strings in one process. The advanced
logger batches binary records from worker threads or processes into shared
memory and dispatches them from a master. Both provide file, Discord and
Telegram handlers.

## Standard logger

```python
from mm_toolbox.logging.standard import FileLogHandler, Logger, LoggerConfig, LogLevel

config = LoggerConfig(
    base_level=LogLevel.INFO,
    flush_on_interval=True,
    flush_interval_s=0.5,
    do_stdout=True,
)
with Logger("app", config, [FileLogHandler("app.txt", create=True)]) as logger:
    logger.info("booted")
    logger.error("something went wrong")
```

Messages flush when the configured size threshold is reached, when a later log
call observes an elapsed interval, or when `flush()` is called explicitly.
Interval checks use the internal monotonic nanosecond clock; they do not start
a background timer. A quiet logger needs an explicit flush to meet an external
schedule.

The standard logger always registers an interpreter-exit shutdown hook.
`shutdown()` and context-manager exit flush pending messages and close handlers,
including a bounded drain of remote deliveries. Explicit shutdown unregisters
the exit hook and repeated shutdown is harmless.

Severity filtering compares cached integer values. `logger.set_log_level(level)`
and `config.set_base_level(level)` refresh that cache. Assignment to
`config.base_level` does so too, keeping loggers sharing a configuration in sync.
Invalid levels raise rather than silently selecting INFO.

Local formatting and file writes run in the caller's thread. Remote handlers
queue deliveries on their own background workers so logging from an active
asyncio loop does not block it on HTTP. The logger's message buffer itself is
not synchronized for concurrent callers.

## Advanced logger

```python
from mm_toolbox.logging.advanced import (
    FileLogHandler, LoggerConfig, LogLevel, MasterLogger, WorkerLogger,
)

config = LoggerConfig(base_level=LogLevel.INFO, do_stdout=False)
master = MasterLogger(config, [FileLogHandler("workers.txt", create=True)])
worker = WorkerLogger(config, name="DataProcessor")
try:
    worker.info(b"Processing batch 123")
    worker.error(b"Failed to process item")
finally:
    worker.shutdown()
    master.shutdown()
```

Pass the same configuration, or at least the same shared-memory path and
transport settings, to each process. An omitted path is randomized; separately
constructed default configurations do not automatically discover each other.

Each worker accepts UTF-8 message bytes, timestamps events with the internal
wall clock, and flushes batches on its interval or message/byte limits. Only one
active worker is allowed per thread. Worker batch access is protected by a lock.
The master owns a drain thread and routes completed batches to its handlers.
`do_stdout=True` mirrors both worker messages and direct master messages on the
master's stdout.

Useful configuration limits:

- `flush_interval_s`: finite positive worker flush interval, default 1 second.
- `max_batch_messages`: message threshold, default 10,000.
- `max_batch_bytes`: byte threshold, default 1 MiB.
- `shm_capacity_bytes`: transport capacity, default 64 MiB.
- `emit_internal`: optional internal lifecycle messages, default false.

The producer shard count follows the capacity and batch limit, up to 64 shards
including the master. Defaults provide 32 shards and room for 31 workers.
Records that cannot fit the shard budget are rejected and reported.

The transport publishes complete binary frames. The decoder validates framing,
lengths, message type and severity. A malformed frame is reported to stderr and
skipped without discarding subsequent valid frames in the same drain.
Heartbeat and DATA message types are reserved; health monitoring and automatic
reconnection are not implemented.

`PyLog` contains a timestamp, name, severity and message. Its byte fields accept
`bytes` or `memoryview`. Decoded messages retain readonly views into their owned
Python batch bytes; formatting converts those fields to text as needed. This
avoids an extra message-field copy during decoding, rather than making the
entire shared-memory transport zero-copy.

## Shared handler delivery policy

Both logging stacks use the same HTTP implementation and policy:

- Up to three attempts in a ten-second total budget, including rate waits,
  requests and backoff. Each HTTP request has a three-second session timeout.
- Retry HTTP 429, HTTP 5xx and transient connection/time-out failures. Other
  client errors fail immediately. Ordinary backoff is 0.1 then 0.2 seconds.
- Honor numeric or HTTP-date Retry-After values, including on HTTP 503. A delay
  beyond the remaining budget fails explicitly instead of retrying early.
- Use the internal fixed-window `RateLimiter`: five Discord requests per two
  seconds and one Telegram request per second. These are library defaults;
  server Retry-After responses still govern retries.
- Pack small messages together and split oversized ones, using 1,800 characters
  for Discord and 3,500 for Telegram. Telegram disables link previews.
- Limit each standard queue to 256 waiting deliveries and each advanced handler
  to 256 scheduled operations. Saturation and shutdown loss are reported through
  the error callback or stderr.

All elapsed-time and wall-clock reads use the internal time utilities. The
HTTP-date parser remains `email.utils.parsedate_to_datetime`: the internal
parser supports ISO 8601, which is a different date format.

Both stacks reuse one HTTP session per remote handler. The session is created
on first delivery and closed on its owning event loop during handler cleanup.
Reopening a standard handler creates a fresh session on its new delivery loop.
Standard handlers process their delivery queues serially; advanced handlers
schedule concurrent operations. Both apply the same retry, pacing and payload
policies.

## Shutdown and accounting

Handler `close(timeout_s=2.0)` uses one bounded drain deadline. Advanced master
`shutdown(timeout_s=2.0)` shares its remaining deadline across the drain thread
and its handlers. Pending work may be cancelled or discarded at the deadline
and is reported. A synchronous custom handler cannot be forcibly interrupted;
a timed-out master leaves mappings owned by its drain thread until cleanup is
safe. Stop workers before the master to include their final batches.

Both file handlers retain an append handle until close and flush after each
batch. Advanced file handlers serialize batches and close under the lifecycle
lock so concurrent master/drain calls cannot race with file cleanup.

## Benchmark results (2026-11-04)

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `c5f9dba`.
These are local measurements and do not establish performance on other machines.

Both drivers use in-memory counting handlers, so the numbers measure logger
and transport overhead.

### Standard logger

`emit_info_ns_per_msg` times the caller-side `Logger.info()` path in batches of
1,000 calls; it is enqueue/emit latency, not delivery completion. The measured
p50 was 924.5 ns/message (100,000 measured messages, after 1,000 warmups).
After shutdown, all 101,000 attempted messages were delivered in one flush with
zero loss. The separate shutdown p50s were 792 ns with an empty buffer and
1,958 ns while flushing 128 buffered messages (25 samples each).

```sh
.venv/bin/python -u benchmarks/logging/benchmark_standard_logging.py --operations 100000 --warmup 1000 --size 256 --emit-batch-size 1000 --buffer-size 0 --shutdown-batch-size 128 --shutdown-iterations 25 --flush-interval-s 3600
```

### Advanced logger

Five measured repeats and two warmup repeats were run per row. Each duration is
0.5 seconds of worker-active time; startup and drain are excluded from the
reported per-message times and rates. The transport capacity was 64 MiB. The
workloads saturated it, so the table separates attempted and consumed counts
and reports their per-message p50s. Rejections, transport errors, worker errors,
forced exits and drain timeouts were all zero; overwritten batches were counted.

| Sweep / setting | p50 attempted (ns/msg) | p50 consumed (ns/msg) | Attempted | Consumed | Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Message size: 32 B, 4 workers | 100 | 195 | 25,107,232 | 12,801,363 | 12,305,869 (49.01%) |
| Message size: 256 B, 4 workers | 90 | 352 | 27,797,752 | 7,049,342 | 20,748,410 (74.64%) |
| Message size: 2,048 B, 4 workers | 185 | 700 | 13,248,817 | 3,710,848 | 9,537,969 (71.99%) |
| Worker count: 1, 256 B | 323 | 581 | 7,648,377 | 4,253,426 | 3,394,951 (44.39%) |
| Worker count: 2, 256 B | 164 | 344 | 15,094,795 | 7,279,094 | 7,815,701 (51.78%) |
| Worker count: 4, 256 B | 94 | 348 | 25,094,197 | 7,027,868 | 18,066,329 (71.99%) |
| Worker count: 8, 256 B | 71 | 337 | 32,030,441 | 7,018,226 | 25,012,215 (78.09%) |

Per-message p50s are medians over five repeat samples; counts and loss are
aggregated over the five measured runs. This is an overload workload, not a
loss-free delivery result.

```sh
.venv/bin/python -u benchmarks/logging/benchmark_advanced_logging.py --axis both --workers 4 --message-size 256 --message-sizes 32,256,2048 --worker-counts 1,2,4,8 --operations 5 --warmup 2 --duration 0.5 --flush-interval 0.05 --startup-timeout 15 --drain-timeout 5 --ipc-dir /tmp/mm-toolbox-advanced-logging-ipc
```

`worker.get_transport_stats()` returns cumulative `rejected_messages`,
`transport_errors` and `overwritten_batches`, including the final flush and
cleanup. Read it after shutdown for final totals. Overwritten batches can each
contain many records. Diagnostics are sampled while counters remain exact;
these counters do not acknowledge downstream handler delivery.

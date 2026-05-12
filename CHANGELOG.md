# Changelog

All notable changes to this project will be documented in this file.

## 1.0.0b8 (Unreleased)

## 1.0.0b7 (2026-05-12)

### Orderbook
- Lower minimum levels from 64 to 4.
- Optimize hot paths with local variable binding and eliminate nested calls.
- Implement incremental delta maintenance and BBO caching.
- Fix P0 core bugs: BBO worse-price handling, delta bid loop, u64 overflow.
- Fix heap buffer overflow in `consume_deltas`.
- Fix `PyOrderbookLevel.from_struct()` missing `__cinit__` arguments.
- Guard double-to-uint64 conversions against non-finite and negative values.
- Reorganize tests into 3-tier architecture and fix weak/zero assertions.

### Candles
- Fix critical bugs and improve performance.

### Time
- Harden C time module: fix thread safety, remove malloc, add `nogil`, buffer hardening.

### Utilities
- Replace Decimal with pure C implementation for `_rounding_factor`.

### SHM / Ringbuffer
- Fix SPSC ringbuffer bugs and improve performance.
- Split monolithic `core.pyx` into `_shm/spsc/mpsc` modules.
- Add `consume_into`/`consume_all_into` for SHM ringbuffers.
- Remove ZMQ IPC ringbuffer and `zmq` dependency.
- Add ringbuffer protocols; standardize `insert`/`insert_batch` to return `bool`.
- Remove packed operations from SHM ringbuffers.
- Unify monolithic and split ringbuffer APIs.
- Remove `overwrite_latest`; add `insert_char`/`consume_into` to bytes ringbuffers.
- Add allocation overflow checks in `BytesRingBufferFast` constructor.
- Validate message lengths in consumer path to prevent out-of-bounds access.
- Harden shared-memory file creation with `O_EXCL`, `O_NOFOLLOW`, and `fstat` validation.
- Remove `insert_char` from public Python API to prevent out-of-bounds reads.

### Logging
- Finalize advanced logging system.
- Redesign standard logger as single-threaded sync logger.
- Fix `.shm/` artifact buildup in tests by using `tmp_path` for IPC fixtures.
- Create fresh HTTP sessions per push and randomize default SHM path.
- Prevent u32 overflow in binary protocol buffer capacity calculations.

### Websocket
- Remove `WsConnectionState`; add direct accessors and tests.
- Fold `_safe` methods into dispatch; eliminate duplicate state checks.
- Remove redundant state checks from public send methods.
- Remove thread-safety dispatch; assume single-threaded callers.
- Increase default ping interval to 1s; use fast EMA for latency tracking.
- Replace `get_type_hints` with safe annotation inspection to prevent code execution.
- Add local server benchmarks for `WsSingle` and `WsPool`.
- Remove live Binance benchmark in favor of local server benchmarks.
- Simplify benchmark CLI to symbols and stream-kinds only.

### Moving Average
- Refactor API, fix ringbuffer compatibility, add tests and protocol.
- Align TEMA `pxd` declarations with `pyx` implementation.

### Rate Limiter
- Simplify design and fix correctness issues.

### Benchmarks
- Add MPSC, candles, moving average, rate limiter, and rounding benchmarks.
- Migrate websocket benchmarks to core `BenchmarkReporter`.
- Eliminate all artifact leaks across IPC, SHM, and logging tests/benchmarks.

### Examples
- Add candle types, moving averages, and MPSC pipeline demos.
- Restore `binance_stream.py` with SPSC SHM ringbuffer.
- Fix `binance_stream.py` SPSC startup race condition.
- Repair `binance_stream.py` with multiple critical fixes.

### Build / CI
- Build macOS arm64-only wheels; ignore local wheel artifacts.
- Add Cython-aware coverage reporting; merge coverage targets into `test-coverage`.

---

## 1.0.0b5 (2026-02-19)

### Orderbook
- Use touch-anchored terminal impact in standard book.
- Align advanced book impact and depth with touch.
- Optimize standard orderbook delta application and precision handling.
- Optimize tick bookkeeping and add depth accessors.
- Add live Binance validation test for standard orderbook.

### Ringbuffer
- Use bitmask indexing in hot accessors for better performance.

### Websocket
- Improve frame handling and latency tracking.
- Refactor hash-history naming and live tests.
- Add live Binance benchmark suite.

### Candles
- Add optional trade payload storage.

### Moving Average / Weights
- Streamline moving average and EMA computations.
- Optimize geometric_weights generation.

### Time / Utilities
- Optimize ISO8601 C-string decode path.

### Logging
- Reduce advanced logger serialization overhead.

### Benchmarks
- Migrate orderbook benchmarks to shared core runner/reporting.
- Refactor ringbuffer numeric benchmark onto core framework.
- Improve ringbuffer IPC/SHM benchmark methodology and reporting.

### Documentation
- Note parser modules are introduced in v1.1.
- Add performance optimization implementation plan.
- Standardize linux wheel output directory to `dist/`.
- Update project metadata and remove stale notes file.

---

## 1.0.0b4 (2026-02-11)

### Orderbook
- Fix duplicate snapshot ticks in standard orderbook.

---

## 1.0.0b2 (2026-02-10)

### Orderbook
- Bind orderbook structs to C typedef headers.
- Fix orderbook correctness bugs.

### Ringbuffer
- Switch generic and numeric consume methods to FIFO semantics.
- Switch bytes consume semantics to FIFO.
- Fix SHM header Cython type declaration.
- Harden shared ringbuffer SHM attach and packed inserts.

### Websocket
- Preserve websocket callback message order.
- Fix websocket-related bugs.

### Candles
- Expose latest candle for Python-level consumers.
- Fix candles correctness bugs.

### Moving Average
- Fix moving-average correctness bugs.

### Logging
- Address rounding and standard logger bugs.

### Build / CI
- Use setuptools backend for native wheels.
- Add cibuildwheel test timeout guard.
- Relax Linux pointer-type warning to unblock wheel builds.
- Remove Windows from wheel build targets.
- Drop Linux before-all package-manager bootstrap.
- Enforce Linux/macOS-only wheel release targets.

### Documentation
- Adapt `WsSingle` and ringbuffer docs to FIFO consume semantics.
- Update beta release metadata and docs.

---

## 1.0.0b1 (2026-02-09)

### Moving Average
- Add moving average base helpers.

### Candles
- **Breaking**: Rename candle VWAP field.

### Logging / Rate Limiter
- Improve handler error handling.
- Tighten rate limiter behavior and documentation.

### Ringbuffer
- Improve pyi type generics.

### Websocket
- Improve error handling in single-connection client (`single.py`).
- Stabilize websocket test suite.

### Build / CI
- Publish Linux + macOS wheels; drop native CPU flag.
- Update macOS deployment target to 11.0.
- Merge building instructions into contributing guide.

---

## 1.0.0b (2026-01-10)

### Orderbook
- **Breaking**: Swap parameter order in `consume_*` functions (asks before bids).
- Fix advanced orderbook delta batch deletions.
- Always compute ticks/lots for advanced and standard orderbooks; removed optional input flags.
- Ignore deltas that would wipe the opposite side without replacement levels.
- Treat snapshots as full replacements for both ladders.
- Rename `update_bbo` to `consume_bbo` to align naming.
- Make C helper tick/lot conversions resilient to floating-point edge cases.
- Skip zero-lot insertions at the BBO to avoid phantom levels.
- Validate numpy ingestion arrays for length mismatches.
- Fix `OrderbookLevels.from_list` to default `norders` to 1 when omitted.
- Consolidate orderbook tests and expand edge-case coverage (crossing deltas, delete+insert same delta, capacity roll-right).

### Ringbuffer
- Add shared-memory ringbuffer with C helpers and Python bindings.
- Fix SHM header validation and producer cache sync.
- Remove silent overflow truncation in fast bytes ringbuffer inserts.
- Add coverage for oversized fast ringbuffer insertions.
- Refactor bytes/generic/numeric backends.

### Logging
- Refactor standard logger/handlers and add standard logging tests.
- Add advanced logging system (initial).

### Misc / Filters / Limiter
- Reorganize filters into a package.
- Add limiter primitives and config.
- Validate `DataBoundsFilter` thresholds on initialization.
- Expose limiter `RateLimitState` in core and tighten Cython initialization paths.
- Add filter/limiter test coverage for configuration validation, thresholds, burst policy, and refill timing.

### Moving Average
- Remove unused base-class hooks from moving average implementations.

### Time
- Fix ISO8601 roundtrip math.

### Parsers
- Introduce fast JSON/Binance parser modules.

### Benchmarks
- Add benchmark framework (orderbook/ringbuffer) and sample datasets.

### Documentation
- Add beta install instructions.
- Add module README documentation.
- Add a candles component overview and consolidate logging docs under `logging/`.
- Update release notes and contributing make targets.

### Build / CI
- Update version to beta; update dependencies.
- Add `clean-caches` Makefile target for pytest/ruff cache cleanup.
- Fix workflow targets to match Makefile commands.
- Reorder CI steps to build extensions before running tests.
- Upgrade artifact actions from v3 to v4.

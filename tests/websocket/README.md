# WebSocket Testing Suite

This directory contains comprehensive tests for the mm-toolbox WebSocket implementation.

## Test Structure

### Connection tests (`connection/`)

- Configuration and state: `test_config.py`, `test_state_machine.py`
- Callbacks, sends and reconnects: `test_callbacks.py`, `test_send_operations.py`, `test_reconnection.py`
- Compiled Cython behavior: `test_native.py` and its `test_native.pyx` support module

### Single-client tests (`single/`)

- Configuration and lifecycle: `test_config.py`, `test_context_manager.py`
- Message callbacks and iteration: `test_message_callback.py`, `test_async_iteration.py`
- Connection failure and recovery: `test_error_scenarios.py`

### Pool tests (`pool/`)

- Configuration and lifecycle: `test_config.py`, `test_context_manager.py`
- Connection management, sends and eviction: `test_connection_mgmt.py`, `test_send_operations.py`, `test_eviction.py`
- Errors, iteration and hash history: `test_error_scenarios.py`, `test_async_iteration.py`, `test_hash_history.py`

### Integration tests (`integration/`)

- End-to-end and realistic workflows: `test_end_to_end.py`, `test_realistic_workflows.py`

### Live Binance tests (`integration/test_live_binance.py`)

- **Smoke coverage**: BTC futures `@bookTicker` for both `WsSingle` and `WsPool`
- **Strict decoding**: Typed `msgspec` decoders with strict parsing (no permissive fallback)
- **Deterministic validation**: Schema and event freshness checks
- **Pool coverage**: Active connection checks plus strict message validation
- **Benchmark handoff**: Live load/latency benchmarks moved to `benchmarks/websocket/benchmark_live_binance.py`

**⚠️ Note**: Live tests require `--run-live` flag and internet connection

## Running Tests

### Prerequisites
```bash
# Build the Cython extensions first
make build
# or
uv run python setup.py build_ext --inplace
```

### Basic Test Execution
```bash
# Run all websocket tests
uv run pytest tests/websocket/ -v

# Run specific test categories
uv run pytest tests/websocket/connection/ -v
uv run pytest tests/websocket/single/ -v
uv run pytest tests/websocket/pool/ -v
uv run pytest tests/websocket/integration/ -v
```

### Live Tests (Require Internet)
```bash
# Run futures smoke tests
uv run pytest tests/websocket/integration/test_live_binance.py --run-live -v -s

# Run all tests including live tests
uv run pytest tests/websocket/ --run-live -v
```

### Live Benchmarks (Require Internet)
```bash
# Run websocket incoming-throughput + latency benchmark
uv run python benchmarks/websocket/benchmark_live_binance.py

# Run a longer sampling window
uv run python benchmarks/websocket/benchmark_live_binance.py --sample-window-s 30
```

### Test Options
- `--run-live`: Enable live tests that require internet connection
- `--live-timeout 60`: Set timeout for live tests (default: 30s)
- `-v`: Verbose output
- `-s`: Show print statements
- `-x`: Stop on first failure
- `--tb=short`: Shorter traceback format

### Performance Testing
```bash
# Run with timing information
uv run pytest tests/websocket/ -v --durations=10

# Run live websocket benchmarks
uv run python benchmarks/websocket/benchmark_live_binance.py
```

## Test Coverage

The test suite provides comprehensive coverage of:

### ✅ **Functional Testing**
- All public APIs and methods
- Configuration validation and edge cases
- State management and transitions
- Data sending and receiving
- Connection lifecycle management
- Pool connection management and eviction

### ✅ **Error Handling**
- Network failures and timeouts
- Invalid configurations and data
- Callback exceptions and isolation
- Resource cleanup and memory management
- Concurrent operation safety

### ✅ **Integration Testing**
- Component interaction and data flow
- Real-world usage patterns
- Live stream validation with Binance
- Performance and scalability characteristics

### ✅ **Edge Cases**
- Extreme values and boundary conditions
- Rapid connect/disconnect cycles
- Large message handling
- Thread safety and cleanup
- Configuration corruption recovery

## Live Test Details

### Binance Streams Tested
- **Smoke stream**: `btcusdt@bookTicker` (futures)

### Validation Performed
- Strict `msgspec` schema validation
- Price/quantity numeric and invariant checks
- Timestamp freshness and clock-sanity checks
- Connection stability for single and pool wrappers

### Deterministic Pass Criteria
- No decode failures with strict typed decoders
- Event timestamps remain fresh relative to local receive time
- Required field-level invariants remain valid

### Live Benchmark Details
- **Benchmark script**: `benchmarks/websocket/benchmark_live_binance.py`
- **Combined load streams**: `btcusdt`, `ethusdt`, `solusdt` on `@bookTicker` and raw `@trade`
- **Metrics**: incoming data throughput-per-second percentiles and side-by-side event-latency stats for `WsSingle`, `WsPool`, and `single_vs_pool`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Cython extensions are built
   ```bash
   make build
   ```

2. **Live Test Failures**: Check internet connection and Binance API status
   ```bash
   # Test with single connection first
   uv run pytest tests/websocket/integration/test_live_binance.py::test_single_btc_bookticker_smoke --run-live -v -s
   ```

3. **Thread Cleanup Warnings**: Some tests may show thread cleanup warnings - this is normal for daemon threads

4. **Timeout Issues**: Increase timeout for slow connections
   ```bash
   uv run pytest tests/websocket/ --live-timeout 60 --run-live
   ```

### Debug Mode
```bash
# Run with maximum verbosity and logging
uv run pytest tests/websocket/ -v -s --log-cli-level=DEBUG
```

## Test Performance

- **Unit Tests**: ~30-60 seconds for full suite
- **Integration Tests**: ~15-30 seconds  
- **Live Tests**: ~5-10 minutes (with `--run-live`)
- **Total Coverage**: 95%+ of codebase

The test suite is designed to be both comprehensive and efficient, providing confidence in the WebSocket implementation while maintaining fast development cycles.

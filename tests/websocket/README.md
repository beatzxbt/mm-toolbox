# WebSocket Testing Suite

This directory contains comprehensive tests for the mm-toolbox WebSocket implementation.

## Test Structure

### 1. Configuration Tests (`test_websocket_config.py`)
- **WsConnectionConfig validation**: URL validation, connection ID uniqueness, on_connect messages
- **LatencyTrackerState functionality**: Default creation, custom configuration, latency updates  
- **WsConnectionState properties**: State transitions, latency tracking, message access
- **ConnectionState enum**: Value validation, comparisons, edge cases
- **Edge case handling**: Long URLs, large messages, unicode content

### 2. Connection Tests (`test_websocket_connection.py`)  
- **Basic operations**: Initialization, configuration access, state management
- **Data operations**: Send data/ping/pong with different connection states
- **Frame handling**: Message processing, memory safety, buffer limits
- **Latency tracking**: Ping/pong cycles, timing calculations, thread safety
- **Callback handling**: Connection/disconnection events, error handling
- **Error handling**: Transport failures, callback exceptions, cleanup

### 3. Single Connection Tests (`test_websocket_single.py`)
- **Basic operations**: Initialization, callback validation, state management
- **Connection management**: Configuration updates, data sending, cleanup
- **Async context manager**: Entry/exit handling, connection lifecycle
- **Async iteration**: Message consumption, error handling
- **Start method**: Reconnection logic, message processing loops
- **Error handling**: Invalid callbacks, concurrent operations, edge cases

### 4. Pool Tests (`test_websocket_pool.py`)
- **Pool configuration**: Validation, default creation, edge cases
- **Basic operations**: Initialization, callback handling, state management
- **Connection management**: Fast connection selection, eviction logic
- **Timed operations**: Background thread management, eviction cycles
- **Async context manager**: Pool lifecycle, concurrent connection creation
- **Async iteration**: Message consumption from pool ringbuffer
- **Error handling**: Connection failures, concurrent operations

### 5. Live Binance Tests (`integration/test_live_binance.py`)
- **Smoke coverage**: BTC futures `@bookTicker` for both `WsSingle` and `WsPool`
- **Load coverage**: Combined BTC/ETH/SOL futures streams for `@bookTicker` and raw `@trade`
- **Latency coverage**: Side-by-side `WsSingle` vs `WsPool` latency stats over the same live stream window
- **Strict decoding**: Typed `msgspec` decoders with strict parsing (no permissive fallback)
- **Deterministic validation**: Schema, symbol/stream coverage, and event freshness checks
- **Pool coverage**: Active connection checks plus combined-stream message validation

**⚠️ Note**: Live tests require `--run-live` flag and internet connection

### 6. Error Handling Tests (`test_websocket_errors.py`)
- **Connection errors**: Disconnected operations, transport failures
- **Callback isolation**: Exception handling, error propagation
- **Resource management**: Memory leaks, thread cleanup, buffer overflow
- **Edge cases**: Extreme values, concurrent operations, configuration corruption
- **Network simulation**: Timeout errors, connection resets, unreachable networks
- **Real-world scenarios**: Rapid connect/disconnect, message flooding

### 7. Integration Tests (`test_websocket_integration.py`)
- **Full lifecycle testing**: Complete connection workflows
- **Component interaction**: Single to pool migration, concurrent operations
- **State consistency**: Cross-component state synchronization
- **Performance characteristics**: Scalability, throughput testing
- **Resource cleanup**: Comprehensive cleanup validation
- **Data integrity**: End-to-end message validation

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
uv run pytest tests/websocket/test_websocket_config.py -v
uv run pytest tests/websocket/test_websocket_connection.py -v
uv run pytest tests/websocket/test_websocket_single.py -v
uv run pytest tests/websocket/test_websocket_pool.py -v
uv run pytest tests/websocket/test_websocket_errors.py -v
uv run pytest tests/websocket/test_websocket_integration.py -v
```

### Live Tests (Require Internet)
```bash
# Run futures smoke tests (live, non-stress)
uv run pytest tests/websocket/integration/test_live_binance.py -m "live and not stress" --run-live -v -s

# Run futures load/stress tests (BTC/ETH/SOL @bookTicker + @trade)
uv run pytest tests/websocket/integration/test_live_binance.py -m "live and stress" --run-live --live-timeout 60 -v -s

# Run side-by-side latency comparison only (single vs pool, same live window)
uv run pytest tests/websocket/integration/test_live_binance.py::test_side_by_side_latency_stats_single_vs_pool --run-live --live-timeout 30 -v -s

# Run all tests including live tests
uv run pytest tests/websocket/ --run-live -v
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

# Run only live load tests
uv run pytest tests/websocket/integration/test_live_binance.py -m "live and stress" --run-live --live-timeout 60 -v -s
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
- **Load streams**: `btcusdt`, `ethusdt`, `solusdt` on `@bookTicker` and raw `@trade`
- **Transport mode**: combined futures stream endpoint for load tests

### Validation Performed
- Strict `msgspec` schema validation
- Price/quantity numeric and invariant checks
- Timestamp freshness and clock-sanity checks
- Required stream coverage across all configured symbols
- Connection stability for single and pool wrappers
- Side-by-side latency summary stats (`mean`, `p50`, `p90`, `p99`, `min`, `max`)

### Deterministic Pass Criteria
- No decode failures with strict typed decoders
- All expected streams are observed within timeout
- Event timestamps remain fresh relative to local receive time
- Required field-level invariants remain valid per event type

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

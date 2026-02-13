# Performance Optimization Plan — mm-toolbox

## Context

The TODO.md lists 21 performance improvements across 7 subsystems. This plan covers all of them (deferring 2 as noted), ordered by safety/impact/dependency. Each optimization has been verified against the actual codebase to ensure correctness and no unintended side-effects.

**Deferred items (by user decision):**
- Candles C-struct accumulation (TODO item 2) — revisit after measuring deepcopy elimination impact
- MPSC via SPSC SHM rings (logging TODO item 5) — separate project
- Moving average `update_batch()` (TODO item 3 partial) — excluded from this pass
- Rounding in-place vector APIs `bids_into`/`asks_into`/`sizes_into` — excluded from this pass
- BytesRingBuffer hash index for uniqueness checks — excluded from this pass

---

## Phase 1: Low-Risk Isolated Changes

### 1.1 — Vectorize `ema_weights`

**File:** `src/mm_toolbox/weights/ema.py`

Replace list comprehension with vectorized NumPy:
```python
# Before (line 28-31):
weights = np.array(
    [alpha * (1.0 - alpha) ** i for i in range(window - 1, -1, -1)],
    dtype=np.float64,
)

# After:
exponents = np.arange(window - 1, -1, -1, dtype=np.float64)
weights = alpha * np.power(1.0 - alpha, exponents)
```

**Safety:** `np.power` with float64 produces identical results to Python `**`. Output shape/dtype unchanged.
**Risk:** None. Pure computation replacement.

---

### 1.2 — Filter by base log level in worker before serialization

**File:** `src/mm_toolbox/logging/advanced/worker.pyx`

The config already stores `base_level` as `CLogLevel` (config.pyx:33), but the worker never checks it. Add a guard in each of the 5 log methods (trace/debug/info/warning/error):

```cython
# Before (e.g. trace, line 117):
if self._is_running:

# After:
if self._is_running and CLogLevel.TRACE >= self._config.base_level:
```

Pattern: `CLogLevel.TRACE >= base_level` for trace, `CLogLevel.DEBUG >= base_level` for debug, etc. Since `CLogLevel` is an enum (TRACE=0, DEBUG=1, INFO=2, WARNING=3, ERROR=4), the comparison `level >= base_level` correctly filters.

**Safety:** Default `base_level` is INFO. This means TRACE and DEBUG are now filtered at the worker, skipping `.encode('utf-8')` and `_add_log_to_batch` serialization entirely. When `base_level=TRACE` (0), all messages pass (identical to current behavior).
**Risk:** None. Purely additive filtering.

---

### 1.3 — Remove per-log worker name (use batch header name)

**File:** `src/mm_toolbox/logging/advanced/worker.pyx`

Currently `_add_log_to_batch()` (lines 103-104) writes `name_len + name_bytes` for **every log entry**, despite `_flush_logs()` (lines 78-79) already writing the worker name in the batch header. All logs in a single batch come from the same worker.

**Change:** Remove the per-log name writes from `_add_log_to_batch()`. In the master's `_decode_worker_message()`, use the batch-level name for all logs in the batch.

**Files to modify:**
- `src/mm_toolbox/logging/advanced/worker.pyx` — remove lines 103-104 from `_add_log_to_batch`
- `src/mm_toolbox/logging/advanced/master.pyx` — in `_decode_worker_message`, use the batch-level worker name (already parsed at lines 74-75) instead of per-log name (lines 91-92)

**Safety:** Wire format changes, but both sides are updated atomically. Saves `(4 + name_len)` bytes per log entry.
**Risk:** Low — both worker and master must be rebuilt together. No backward compat issue since these are in-process components.

---

### 1.4 — Cache `inv_tick_size` / `inv_lot_size` for tick computation

**Files:**
- `src/mm_toolbox/orderbook/standard/orderbook.py`
- `src/mm_toolbox/orderbook/standard/level.py`

**Changes:**
1. In `Orderbook.__init__`, compute `self._inv_tick_size = 1.0 / tick_size` and `self._inv_lot_size = 1.0 / lot_size`.
2. Add `price_to_ticks_fast(price, inv_tick_size) -> int` returning `int(price * inv_tick_size)`.
3. Add optional `inv_tick_size`/`inv_lot_size` params to `OrderbookLevel.add_precision_info()`. When provided, use multiplication instead of division (skip `math.floor` call).
4. Update all call sites in `orderbook.py` to pass inverse values.

**Safety:** For non-negative prices (validated by `OrderbookLevel.__post_init__`: `price >= 0.0`), `int(price * inv)` equals `int(floor(price / tick_size))` because `int()` truncates toward zero which matches `floor()` for non-negative values. Both share identical floating-point behavior.
**Risk:** None for non-negative prices. Orderbook prices are always >= 0.

---

## Phase 2: Orderbook Sorting Optimizations

**These are interconnected. Implement 2.1 first, then 2.2 and 2.3.**

### 2.1 — Incremental sorted tick lists with `bisect`

**File:** `src/mm_toolbox/orderbook/standard/orderbook.py`

**Core change:** Replace `list.sort()` after every mutation with `bisect.insort()` for insertions and `bisect`-based removal for deletions.

**Bid tick direction change (confirmed by user):** Store bids **ascending** (like asks). Best bid = `[-1]` instead of `[0]`.

**Detailed changes:**

1. **Import:** `from bisect import insort, bisect_left`

2. **`consume_snapshot()`** — Keep bulk `.sort()` here (rebuilding from scratch, single sort is appropriate).
   - Asks: `self._sorted_ask_ticks.sort()` (unchanged, already ascending)
   - Bids: `self._sorted_bid_ticks.sort()` (remove `reverse=True` — now ascending)

3. **`consume_deltas()`** — Replace append + sort with insort:
   ```python
   # Asks: replace append + sort
   insort(self._sorted_ask_ticks, ticks)  # instead of append + sort()
   # Remove trailing self._sorted_ask_ticks.sort()

   # Bids: same pattern
   insort(self._sorted_bid_ticks, ticks)  # instead of append + sort(reverse=True)
   # Remove trailing self._sorted_bid_ticks.sort(reverse=True)
   ```

4. **`_maybe_remove_ask/bid()`** — Replace `list.remove(ticks)` (O(n) scan + O(n) shift) with:
   ```python
   idx = bisect_left(self._sorted_ask_ticks, ticks)
   if idx < len(self._sorted_ask_ticks) and self._sorted_ask_ticks[idx] == ticks:
       del self._sorted_ask_ticks[idx]
   ```

5. **Best-bid access refactor** — Update all `self._sorted_bid_ticks[0]` to `self._sorted_bid_ticks[-1]`:
   - `consume_bbo()` lines 163, 166, 172
   - `get_bbo()` line 228
   - `get_bbo_spread()` line 236
   - `get_mid_price()` line 244
   - `get_wmid_price()` line 251
   - `does_bbo_price_change()` line 350
   - `does_bbo_cross()` line 359
   - Bid iteration in `iter_bids()` and `get_bids()` — use `reversed()` to maintain highest-first output

6. **`pop(0)` → `pop()`** — All `self._sorted_bid_ticks.pop(0)` (O(n)) become `self._sorted_bid_ticks.pop()` (O(1)).

**Safety:** `bisect.insort` maintains sorted order if list is already sorted. After `consume_snapshot` sorts once, all subsequent `insort` calls maintain the invariant. No duplicate ticks enter because dict-membership is checked first.
**Risk:** Medium — touching ~12 access points for the bid direction change. Mitigate by adding helper properties:
```python
@property
def _best_bid_ticks(self) -> int:
    return self._sorted_bid_ticks[-1]

@property
def _best_ask_ticks(self) -> int:
    return self._sorted_ask_ticks[0]
```

---

### 2.2 — Fast-path `consume_bbo` without sorting

**File:** `src/mm_toolbox/orderbook/standard/orderbook.py`

After 2.1, the sort calls in `consume_bbo` are replaced by `insort`. Add a fast-path: if the updated tick already exists in the dict (just a size/price update at the same tick level), skip the sorted-list operation entirely:

```python
if bid_ticks in self._bids:
    self._bids[bid_ticks] = bid  # Size update only, tick already in sorted list
else:
    # New tick — needs insort + possibly evict old best
    ...
    insort(self._sorted_bid_ticks, bid_ticks)
```

**Safety:** Natural consequence of 2.1. The "tick already exists" case requires zero sorted-list operations.
**Risk:** Low — the existing removal logic for old best bid (lines 171-175) must be preserved for the "new tick" path.

---

### 2.3 — Optional `depth` argument for accessors

**File:** `src/mm_toolbox/orderbook/standard/orderbook.py`

```python
from itertools import islice

def get_asks(self, depth: int | None = None) -> list[OrderbookLevel]:
    self._ensure_populated()
    ticks = self._sorted_ask_ticks[:depth] if depth is not None else self._sorted_ask_ticks
    return [self._asks[tick] for tick in ticks]

def get_bids(self, depth: int | None = None) -> list[OrderbookLevel]:
    self._ensure_populated()
    if depth is not None:
        ticks = self._sorted_bid_ticks[-depth:]  # ascending, last N are best
        return [self._bids[tick] for tick in reversed(ticks)]
    return [self._bids[tick] for tick in reversed(self._sorted_bid_ticks)]

def iter_asks(self, depth: int | None = None) -> Iterator[OrderbookLevel]:
    self._ensure_populated()
    ticks = islice(self._sorted_ask_ticks, depth) if depth is not None else self._sorted_ask_ticks
    for tick in ticks:
        yield self._asks[tick]

def iter_bids(self, depth: int | None = None) -> Iterator[OrderbookLevel]:
    self._ensure_populated()
    if depth is not None:
        start = max(0, len(self._sorted_bid_ticks) - depth)
        ticks = self._sorted_bid_ticks[start:]
    else:
        ticks = self._sorted_bid_ticks
    for tick in reversed(ticks):
        yield self._bids[tick]
```

**Safety:** Default `depth=None` preserves existing behavior. Purely additive API.
**Risk:** Very low. Test with depth=0, 1, len, >len.

---

## Phase 3: Candle Optimization

### 3.1 — Config to disable per-trade storage + eliminate `copy.deepcopy`

**Files:**
- `src/mm_toolbox/candles/base.pyx` + `.pxd`
- `src/mm_toolbox/candles/time.pyx`, `tick.pyx`, `price.pyx`, `volume.pyx`, `multi.pyx`

**Changes:**

1. Add `cdef bint _store_trades` to `BaseCandles` in `.pxd`.

2. Add `bint store_trades=True` param to `BaseCandles.__init__()`:
   ```python
   self._store_trades = store_trades
   ```

3. Add shallow copy method to `Candle`:
   ```python
   def copy(self, bint include_trades=True) -> Self:
       if include_trades:
           return copy.deepcopy(self)
       return Candle(
           open_time_ms=self.open_time_ms,
           close_time_ms=self.close_time_ms,
           open_price=self.open_price,
           high_price=self.high_price,
           low_price=self.low_price,
           close_price=self.close_price,
           buy_size=self.buy_size,
           buy_volume=self.buy_volume,
           sell_size=self.sell_size,
           sell_volume=self.sell_volume,
           vwap=self.vwap,
           num_trades=self.num_trades,
           trades=[],
       )
   ```

4. In `insert_and_reset_candle()`:
   ```python
   cdef object closed_candle = self.latest_candle.copy(include_trades=self._store_trades)
   ```

5. In all 5 candle subtype `process_trade()` methods, guard the trades append:
   ```python
   if self._store_trades:
       self.latest_candle.trades.append(trade)
   ```
   This also skips Trade object creation in VolumeCandles/MultiCandles chunking when `_store_trades=False`.

6. Propagate `store_trades` through all subclass `__init__` signatures.

**Safety:** With `store_trades=False`, `.trades` will be `[]` on emitted candles. Consumers must be aware. All scalar fields are copied field-by-field — no mutation risk.
**Risk:** Low. The `deepcopy` bypass is the main performance win. VolumeCandles/MultiCandles also skip creating split-Trade objects, saving further allocation.

---

## Phase 4: Moving Average Optimizations

### 4.1 — Typed memoryviews for `initialize()` loops

**Files:** `src/mm_toolbox/moving_average/sma.pyx`, `wma.pyx`

Add strided memoryview declaration (using `double[:]` not `double[::1]` for safety with non-contiguous arrays):

```cython
# SMA.initialize (sma.pyx):
cpdef double initialize(self, cnp.ndarray values):
    cdef:
        int     i, n = values.shape[0]
        double  raw_value
        double[:] vals = values  # Typed memoryview
    ...
    for i in range(0, n):
        raw_value = vals[i]  # was: values[i]
```

Same pattern for WMA. EMA/TEMA already call `self.update()` per element — no loop to optimize.

**Safety:** `double[:]` accepts both contiguous and strided arrays. If the array isn't float64, Cython raises a `ValueError` at the memoryview assignment — correct fail-fast behavior.
**Risk:** Very low.

---

### 4.2 — Pass `disable_async=True` to `NumericRingBuffer` when `is_fast=True`

**File:** `src/mm_toolbox/moving_average/base.pyx`

```python
# Before (line ~45):
self._values = NumericRingBuffer(window, dtype=np.dtype(np.float64))

# After:
self._values = NumericRingBuffer(window, dtype=np.dtype(np.float64), disable_async=is_fast)
```

**Safety:** `disable_async=True` skips `asyncio.Event` allocation inside the ring buffer. The `_values` buffer is never awaited when `is_fast=True` anyway.
**Risk:** Very low. Purely avoids unnecessary asyncio.Event creation.

---

## Phase 5: WebSocket Optimizations

### 5.1 — Replace bytes concatenation with `bytearray` for fragmented frames

**Files:** `src/mm_toolbox/websocket/connection.pyx` + `.pxd`

1. In `.pxd`, change `bytes _unfin_msg_buffer` to `bytearray _unfin_msg_buffer`.
2. In `__cinit__`: `self._unfin_msg_buffer = bytearray()`
3. Accumulation (line 264): `self._unfin_msg_buffer.extend(frame_bytes)` instead of `+=`
4. Insert to ringbuffer (line 271): `self._ringbuffer.insert(bytes(self._unfin_msg_buffer))`
5. Reset: `self._unfin_msg_buffer.clear()` (or `= bytearray()`)

**Safety:** `bytearray.extend()` appends in-place when capacity allows, avoiding O(n) copy on each concatenation. The `bytes()` conversion at insert creates a single immutable copy for the ringbuffer.
**Risk:** Low. No external code holds references to `_unfin_msg_buffer` during accumulation.

---

### 5.2 — Non-fragmented frame fast-path

**File:** `src/mm_toolbox/websocket/connection.pyx`

Add fast-path after control frame handling: if `fin=1` and no pending fragments, insert directly without touching the bytearray buffer:

```cython
# After control frame handling, before accumulation:
if not frame_unfinished and self._unfin_msg_size == 0:
    if frame_msg_type in (WSMsgType.TEXT, WSMsgType.CONTINUATION, WSMsgType.BINARY):
        self._ringbuffer.insert(frame_bytes)
        self._seq_id += 1
        return
```

**Safety:** Non-fragmented frames (the common case) skip the bytearray accumulation path entirely. Single `frame.get_payload_as_bytes()` + single ringbuffer insert.
**Risk:** Low. The fast-path condition (`fin=1 and unfin_size=0`) is unambiguous.

---

### 5.3 — Investigate + implement picows auto-ping/auto-pong

**File:** `src/mm_toolbox/websocket/connection.pyx`

**Step 1:** Check if picows `ws_connect` supports `auto_ping_interval` / `auto_ping_timeout` params. Check the installed picows `.pxd` and documentation.

**Step 2 (if supported):**
1. Pass `auto_ping_interval=10.0, auto_ping_timeout=30.0` to `ws_connect` in `WsConnection.new()`.
2. Remove `_timed_operations_thread` and `_timed_operations` method.
3. Keep PONG handling in `on_ws_frame` for latency tracking — picows auto-ping still triggers PONG responses that we can timestamp.
4. Simplify `close()` — no thread to manage.

**Step 3 (if not supported):** Skip this optimization gracefully.

**Safety:** Removes an entire daemon thread per connection. Latency tracking continues via frame callbacks.
**Risk:** Medium — depends on picows API. If supported, the change simplifies the code significantly.

---

### 5.4 — Investigate `send_reuse_external_bytearray()` for outbound frames

**File:** `src/mm_toolbox/websocket/connection.pyx`

**Step 1:** Check if picows exposes `send_reuse_external_bytearray()` in its `.pxd`.

**Step 2 (if supported):** Add a new method `send_data_bytearray(bytearray msg)` that uses the reuse API. Keep existing `send_data(bytes msg)` for backward compatibility.

**Step 3 (if not supported):** Skip.

**Safety:** Outbound path is rarely the bottleneck. Low priority.
**Risk:** Medium — requires understanding picows headroom requirements.

---

## Phase 6: Logging Flush Optimization

### 6.1 — Collapse worker flush into single buffer

**File:** `src/mm_toolbox/logging/advanced/worker.pyx`

Currently `_flush_logs()` creates a `data_writer`, finalizes to chars, wraps in `InternalMessage` via `create_internal_message()`, then calls `internal_message_to_bytes()` which creates a *second* BinaryWriter and produces another bytes copy.

**Change:** Write InternalMessage header + data payload in a single BinaryWriter:

```cython
cdef void _flush_logs(self):
    if self._num_pending_logs == 0:
        return

    cdef u64 ts = time_ns()
    cdef u32 batch_data_len = 8 + self._len_name + self._batch_writer.length()

    # Single writer for everything
    cdef BinaryWriter full_writer = BinaryWriter(1 + 8 + 4 + batch_data_len)
    full_writer.write_u8(<u8>MessageType.LOG)       # InternalMessage header
    full_writer.write_u64(ts)
    full_writer.write_u32(batch_data_len)
    full_writer.write_u32(self._len_name)            # Data payload
    full_writer.write_chars(self._name_as_chars, self._len_name)
    full_writer.write_u32(self._num_pending_logs)
    full_writer.write_chars(self._batch_writer._buffer, self._batch_writer.length())

    self._transport.insert(full_writer.finalize())
    self._batch_writer.reset()
    self._num_pending_logs = 0
```

Eliminates: one BinaryWriter allocation, `finalize_to_chars()`, `create_internal_message()`, and `internal_message_to_bytes()`.

**Safety:** Wire format is unchanged — master receives the same byte sequence. Only the serialization path is simplified.
**Risk:** Low — actually simpler than the current two-step approach.

---

### 6.2 — Decode with `memoryview` slices (low priority)

**File:** `src/mm_toolbox/logging/advanced/master.pyx`

Replace `BinaryReader.read_bytes()` (which creates `bytes` slices) with `memoryview` slices that defer bytes materialization until handlers need it.

**Safety:** `memoryview` slices of `bytes` don't copy data. Valid as long as source `bytes` is alive.
**Risk:** Low, but micro-optimization. Only implement if profiling shows `_decode_worker_message` as bottleneck.

---

## Implementation Order

| #  | Item | Est. Impact | Risk | Depends On |
|----|------|-------------|------|------------|
| 1  | 1.1 — Vectorize ema_weights | Med | Very Low | — |
| 2  | 1.2 — Worker level filtering | High | Very Low | — |
| 3  | 1.4 — Cache inv_tick_size | Med | Low | — |
| 4  | 2.1 — Bisect sorted ticks | High | Medium | — |
| 5  | 2.2 — Fast-path consume_bbo | Med | Low | 2.1 |
| 6  | 2.3 — Depth args for accessors | Low | Very Low | 2.1 |
| 7  | 3.1 — Disable trade storage | High | Low | — |
| 8  | 4.1 — Typed memoryviews for MA | Med | Low | — |
| 9  | 4.2 — disable_async for fast MA | Low | Very Low | — |
| 10 | 5.1 — bytearray for fragments | Med | Low | — |
| 11 | 5.2 — Non-fragmented fast-path | Med | Low | 5.1 |
| 12 | 5.3 — Auto-ping/pong (if supported) | Med | Medium | — |
| 13 | 5.4 — send_reuse_bytearray (if supported) | Low | Medium | — |
| 14 | 6.1 — Collapse flush buffer | Med | Low | — |
| 15 | 1.3 — Remove per-log worker name | Med | Low | — |
| 16 | 6.2 — memoryview decode | Low | Low | — |

---

## Verification

After each phase:
1. `make format` — ensure ruff formatting passes
2. `make typecheck` — ensure ty check passes
3. `make test` — run full test suite, all tests must pass
4. For phases 2 and 3 specifically: add parametrized test variants (e.g., `store_trades=True/False`, `depth=None/1/5`)
5. For phase 5 (picows-dependent): test with a live websocket connection if possible, otherwise verify via unit test mocks

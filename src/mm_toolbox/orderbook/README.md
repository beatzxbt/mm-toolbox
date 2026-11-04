# Orderbook

Two compatible orderbook implementations live here:
- `standard/`: Python-first, easy to integrate.
- `advanced/`: C/Cython-accelerated for high throughput.

Both expose a largely overlapping API so downstream logic can stay stable while
you switch the backing implementation for performance.

## Architecture overview

### Standard (Python)

The standard orderbook stores levels in dictionaries and maintains sorted tick
lists per side.

```
           bids (desc)                 asks (asc)
┌─────────────────────────┐    ┌─────────────────────────┐
│ dict[tick] -> level     │    │ dict[tick] -> level     │
└──────────┬──────────────┘    └──────────┬──────────────┘
           │                               │
           ▼                               ▼
  sorted bid ticks                  sorted ask ticks
```

### Advanced (C/Cython)

The advanced orderbook stores levels in contiguous, aligned arrays and performs
O(n) merges with tight inner loops.

```
 bids array (desc)                     asks array (asc)
┌─────────────────────────┐    ┌─────────────────────────┐
│ [ticks, lots, norders]  │    │ [ticks, lots, norders]  │
└─────────────────────────┘    └─────────────────────────┘
```

## Standard orderbook

### Characteristics

- Pure Python structures; minimal setup.
- Easy to inspect and debug.
- Best for moderate update rates and small-to-medium depth.

### Flow

```
snapshot/delta/bbo
        │
        ▼
dict + sorted tick lists
        │
        ▼
query helpers (bbo, spread, iterators)
```

## Advanced orderbook

### Characteristics

- C/Cython implementation for high throughput.
- Contiguous memory and linear-time delta merges.
- Optional zero-copy NumPy views for downstream analytics.

### Flow

```
snapshot/delta/bbo
        │
        ▼
normalize raw values + infer/validate sortedness
        │
        ▼
contiguous arrays + memmove merges
        │
        ▼
numpy views / accessors
```

## Choosing between them

Use `standard/` when:
- You want simple Python APIs and minimal build complexity.
- Update rates are modest and readability matters.

Use `advanced/` when:
- You process large tick streams or require strict latency.
- You can provide sorted inputs to unlock linear-time merges.
- You want zero-copy NumPy views and fewer allocations.

## API alignment and differences

Both implementations provide matching read-side helpers:
- `get_asks()`, `get_bids()`
- `iter_asks()`, `iter_bids()`
- `get_bbo()`, `get_bbo_spread()`, `get_mid_price()`, `get_wmid_price()`
- `get_volume_weighted_mid_price(size, is_base_currency=True)`
- `get_price_impact(size, is_buy, is_base_currency=True)`
- `get_size_for_price_impact_bps(impact_bps, is_buy, is_base_currency=True)`
- `does_bbo_price_change(bid_price, ask_price)`
- `is_bbo_crossed(bid_price, ask_price)`

Ingestion order (both implementations):
- `consume_snapshot(asks, bids)`
- `consume_deltas(asks, bids)`
- `consume_bbo(ask, bid)`

Advanced-only helpers:
- `get_bids_numpy()`, `get_asks_numpy()`
- `consume_snapshot_numpy`, `consume_deltas_numpy`
- `clear()`
- Raw level constructors: `OrderbookLevel`, `OrderbookLevels.from_list`,
  `OrderbookLevels.from_numpy`

## Quick start

### Standard

```python
from mm_toolbox.orderbook.standard import Orderbook

ob = Orderbook(tick_size=0.01, lot_size=0.001, size=500)
ob.consume_snapshot(asks, bids)
ob.consume_deltas(asks_delta, bids_delta)
best_bid, best_ask = ob.get_bbo()
```

### Advanced

```python
from mm_toolbox.orderbook.advanced import AdvancedOrderbook

ob = AdvancedOrderbook(tick_size=0.01, lot_size=0.001, num_levels=1000)
ob.consume_snapshot(asks, bids)
ob.consume_deltas(asks_delta, bids_delta)
best_bid, best_ask = ob.get_bbo()
```

**Note:** The advanced orderbook must be imported from `mm_toolbox.orderbook.advanced`.

## Behavior notes

- Public advanced levels are raw `price`, `size`, and `norders` values.
- Ticks and lots are computed on ingest and stored only in compact internal entries.
- For snapshots and deltas, duplicate normalized price ticks use the final
  original entry's absolute quantity and order count. Zero lots delete a level;
  snapshot zero lots are omitted. Duplicate resolution happens before reversing
  a sorted side, and before applying the advanced capacity limit.
- Snapshots reject crossed books after normalization and leave the previous
  book intact on rejection.
- BBO updates validate raw inputs and reject crossed incoming BBO values.
- `get_mid_price()` and `get_wmid_price()` intentionally floor the result to a
  whole price tick. An odd sum of best bid/ask ticks therefore rounds down.
- `get_volume_weighted_mid_price()` computes each side's average execution price,
  consuming only the needed quantity at the final level, then takes their
  arithmetic midpoint without flooring to ticks. Quote notional is converted
  to base quantity using `get_mid_price()`. Insufficient depth returns infinity;
  nonpositive size returns `get_mid_price()`.
- Advanced `get_price_impact(size, is_buy, is_base_currency)` is touch-anchored and
  returns terminal impact: `abs(last_touched_price - touch_anchor_price)`.
- Advanced `get_size_for_price_impact_bps(impact_bps, is_buy, is_base_currency)`
  measures depth from touch (`best_ask` for buys, `best_bid` for sells) and includes
  boundary levels.
- Deltas apply both sides of a batch before removing crossed levels. Ask updates
  remove crossed bids first; bid updates then remove crossed asks. This avoids
  deleting levels because of a temporary cross within a valid batch.
- Deltas ignore a special edge case: if a delta would wipe the entire
  opposite side and provides no replacement levels, the delta is ignored.

## Performance notes

- Standard: sorting on every update can dominate runtime at high depth.
- Advanced: linear merges are fast but require stricter input ordering.
- Advanced `UNKNOWN` sortedness lazily infers stable input ordering and rejects
  unsorted inputs instead of sorting them on every ingest.

## Benchmark results (2026-11-04)

Both drivers replayed the checked-in 100,000-row fixture. The snapshot initializes
the book, then 1,000 updates warm up the runner; 98,999 updates were measured
(90,019 BBO and 8,980 delta operations). The table reports p50 nanoseconds per
input level. Advanced used 2,048 levels with sortedness hints disabled (`UNKNOWN`).

| Implementation | BBO p50 (ns/level) | Delta p50 (ns/level) | Measured updates/s |
| --- | ---: | ---: | ---: |
| Standard | 708.5 | 1,019.2 | 84,991 |
| Advanced | 62.5 | 307.5 | 336,305 |

Measured on CPython 3.14.4, NumPy 2.5.4, Cython 3.3.0, Apple clang 21.0.0,
macOS 26.6.2 arm64, Apple M2 (8 CPU cores, 16 GiB), at revision `c5f9dba`.
These are local measurements and do not establish performance on other machines.

Reproduce from the repository root:

```sh
.venv/bin/python -u benchmarks/orderbook/benchmark_standard_orderbook.py --input benchmarks/orderbook/data/btcusdt_100k.jsonl --operations 100000 --warmup 1000
.venv/bin/python -u benchmarks/orderbook/benchmark_advanced_orderbook.py --input benchmarks/orderbook/data/btcusdt_100k.jsonl --operations 100000 --warmup 1000 --levels 2048
```

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
│ [tick, price, size, ...]│    │ [tick, price, size, ...]│
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
normalize (sortedness hints)
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
- `does_bbo_cross(bid_price, ask_price)`

Ingestion order (both implementations):
- `consume_snapshot(asks, bids)`
- `consume_deltas(asks, bids)`
- `consume_bbo(ask, bid)`

Advanced-only helpers:
- `get_bids_numpy(depth=None)`, `get_asks_numpy(depth=None)`
- Buffer constructors: `create_orderbook_level*`, `create_orderbook_levels_from_list/numpy`

## Quick start

### Standard

```python
from mm_toolbox.orderbook.standard import Orderbook

ob = Orderbook()
ob.consume_snapshot(asks, bids)
ob.consume_deltas(asks_delta, bids_delta)
best_bid, best_ask = ob.get_bbo()
```

### Advanced

```python
from mm_toolbox.orderbook.advanced import Orderbook

ob = Orderbook()
ob.consume_snapshot(asks, bids)
ob.consume_deltas(asks_delta, bids_delta)
best_bid, best_ask = ob.get_bbo()
```

## Behavior notes

- Ticks/lots are always computed on ingest; pre-filled values are overwritten.
- Snapshots replace state without validation; callers must supply non-crossed data.
- BBO updates assume non-crossing inputs; no validation is performed.
- Advanced `get_price_impact(size, is_buy, is_base_currency)` is touch-anchored and
  returns terminal impact: `abs(last_touched_price - touch_anchor_price)`.
- Advanced `get_size_for_price_impact_bps(impact_bps, is_buy, is_base_currency)`
  measures depth from touch (`best_ask` for buys, `best_bid` for sells) and includes
  boundary levels.
- Advanced deltas ignore a special edge case: if a delta would wipe the entire
  opposite side and provides no replacement levels, the delta is ignored.

## Performance notes

- Standard: sorting on every update can dominate runtime at high depth.
- Advanced: linear merges are fast but require stricter input ordering.
- If you can guarantee sortedness, advanced normalization is a cheap pass.

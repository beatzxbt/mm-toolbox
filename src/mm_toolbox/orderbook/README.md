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
- Snapshots reject crossed books.
- BBO updates validate raw inputs and reject crossed incoming BBO values.
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
- Advanced `UNKNOWN` sortedness lazily infers stable input ordering and rejects
  unsorted inputs instead of sorting them on every ingest.

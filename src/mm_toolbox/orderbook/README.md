# Orderbook implementations

Two compatible orderbook designs live here: a simple Python version in `standard/` and a high‑performance Cython version in `advanced/`. They expose a largely overlapping API so you can switch based on your performance needs without rewriting downstream logic.

## standard/ — simple, Python-first
- Designed for small systems, prototypes, and easy integration.
- Pure Python data structures: `dict[int, OrderbookLevel]` plus sorted tick lists.
- Updates run in straightforward Python with minimal setup; precision info (ticks/lots) can be auto-filled on ingest.
- Great when readability and simplicity matter more than maximum throughput.

How it works (briefly):
- Snapshots build dicts for bids/asks and then sort tick lists.
- Deltas update or remove levels and re-sort the affected side(s).
- Best for moderate depth and update rates where Python sorting and dict operations are acceptable.

## advanced/ — C/Cython-accelerated
- Built for very high throughput and low latency.
- Uses contiguous, aligned C arrays per side and O(n) single-pass updates with tight inner loops and `memmove` insertions.
- Exploits known sortedness of incoming data:
  - You can declare sortedness for snapshots and deltas at construction.
  - When sortedness is known, normalization is a cheap reverse-or-pass; when unknown, a one-time in-place sort occurs.
- Zero-copy views: exposes NumPy-structured arrays for bids/asks when you need vectorized post-processing.
- Requires slightly more setup (construct `OrderbookLevels`, indicate whether levels already include ticks/lots).

How it works (briefly):
- Maintains bids descending and asks ascending in contiguous memory.
- Deltas are merged by walking existing levels once, inserting/removing via `memmove`, and early-breaking outside the current best/worst tick bounds.
- Minimizes Python<->C crossings; most work happens in `nogil` C loops.

## Choosing between them
- Use `standard/` when:
  - You want dead-simple Python APIs and quick iteration.
  - Update rates are modest and you value readability over raw speed.
- Use `advanced/` when:
  - You process large tick streams and need strict latency.
  - You can supply (or assume) sorted inputs to unlock linear-time merges.
  - You benefit from NumPy views and fewer allocations.

## API overlap and small differences

Core ingestion (note parameter order and naming):
- Standard:
  - `consume_snapshot(bids, asks)`
  - `consume_deltas(bids, asks)`
  - `consume_bbo(bid, ask)`       ← order is bids, then asks
- Advanced:
  - `consume_snapshot(asks, bids)`     ← order is asks, then bids
  - `consume_deltas(asks, bids)`       ← order is asks, then bids
  - `consume_bbo(ask, bid)`            ← order reversed

Common accessors/utilities (same names and behavior):
- `get_asks()`
- `get_bids()`
- `iter_asks()`
- `iter_bids()`
- `get_bbo()`
- `get_bbo_spread()`
- `get_mid_price()`
- `get_wmid_price()` (weighted mid)
- `get_volume_average_mid_price(size, is_base_currency=True)` (VAMP)
- `get_price_impact(size, is_buy, is_base_currency=True)`
- `does_bbo_price_change(bid_price, ask_price)`
- `does_bbo_cross(bid_price, ask_price)`

Advanced-only conveniences:
- `get_bids_numpy(depth=None)`, `get_asks_numpy(depth=None)` — zero-copy structured arrays.
- Helpers to build buffers: `create_orderbook_level*`, `create_orderbook_levels_from_list/numpy`, and `free_orderbook_levels`.

Notes:
- Ticks/lots are always computed on ingest; any pre-filled values are overwritten.
- Snapshots replace state without validation; callers are responsible for non-crossed feeds.
- BBO updates assume non-crossing inputs; no validation is performed.
- Extreme edge case (advanced deltas): if a delta would wipe the entire opposite side and provides no replacement levels for it, the delta is ignored.
- If you migrate between implementations, watch the ingestion differences (BBO method name and argument order). The read-side helpers are intentionally aligned to keep the bulk of your code unchanged.

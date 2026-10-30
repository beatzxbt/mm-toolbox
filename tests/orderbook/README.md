# Orderbook correctness tests

The main suite feeds one operation trace to an independent dictionary reference
and three targets: the standard book, the advanced Python API, and the advanced
Cython API. The Cython target is a compiled test adapter that calls the actual
`cdef` methods; it does not route through the advanced Python wrapper.

The reference stores integer price ticks, quantity lots and order counts. Its
input conversion uses Decimal and its calculations use exact rational
arithmetic. It imports no production normalization or orderbook helpers.
Every accepted operation compares all levels, ordering, quantities, order
counts, side lengths and the public BBO getter. Spread, floored midpoint and
weighted midpoint are checked at each populated state. The weighted midpoint
must lie within the normalized BBO, with no tolerance for overshoot. Rejected operations
must preserve state; queries must preserve it too.

## Where the tests live

| File | Responsibility |
| --- | --- |
| `reference.py` / `test_reference.py` | Small dictionary contract and hand-calculated self-checks. |
| `adapters.py` | Input/output adapters for exactly three production targets. |
| `scenarios.py` | Named English behavior cases, numeric cases and deterministic generated traces. |
| `test_scenarios.py` | Shared operation-by-operation comparisons, replay context and calculation checkpoints. |
| `advanced/wrapper_cython/test_wrapper.pyx` | Compiled Cython adapter and focused native delegation tests. |
| `advanced/wrapper_python/test_ingestion_contract.py` | Additional NumPy/scalar conversions, validation, ownership and export lifetimes. |
| `../benchmarks/test_orderbook_collector.py` | Snapshot bridging, cross-feed publication, bounds, failure cleanup and offline feed replay. |
| `advanced/engine/` and `advanced/c/` | Direct native primitives, ring boundaries and memory-sensitive contracts. |

## Behavioral coverage

Each scenario ID is visible in pytest output and a failure includes the target,
capacity, seed, first divergent operation, input and complete expected/actual
rows. One-sided cases have bid and ask variants.

| IDs | Input changes and expected behavior |
| --- | --- |
| S1–S2 | Load a known book, then replace it completely; no previous levels remain. |
| S3–S4 | Repeat prices and zero quantities in snapshots; the last absolute update wins and final zeros disappear before truncation. |
| S5–S6 | All supported sorted input orientations/default inference; crossed or invalid replacements reject atomically without poisoning later ordering. |
| D1–D5 | Absolute quantity/order replacement, insertions, best/interior/worst deletions, absent deletions and empty/one-sided batches. |
| D6–D8 | Duplicate updates, deletion/restoration and a temporary crossing entry later deleted in the same batch. |
| D9–D11 | Completed two-sided moves, partial crossing cleanup and ignored updates that would erase the entire opposite side without replacement. |
| D12 | Equivalent changes across the actual total compacted batch-width boundary; harmless padding cannot change behavior. |
| B1–B4 | Same-price, tighter, one-level and multi-level BBO changes refresh price, quantity and order count. |
| B5–B7 | BBOs in gaps, outside retained depth and across the previous spread retain only valid deeper levels. |
| B8–B10 | Repeated quotes are idempotent; deltas restore only explicitly supplied levels; zero BBOs delete a matching touch and unmatched zeros reject atomically. |
| C1–C4 | Bounded snapshots, eviction without resurrection, worse-price pressure and replacement without accidental eviction. |
| C5 | Cross cleanup and all updates complete before capacity truncation; small and bulk paths retain the same valid tails. |
| C6–C9 | Repeated wrapped heads, clear/reseed, depleted-side recovery and replacement-snapshot recovery. |
| M1–M3 | Spread, floored midpoint, weighted midpoint and independently calculated execution VWAP. |
| M4–M6 | Exact/insufficient depth, fractional requested sizes, base/quote conversion and impact-band boundaries. Unfillable depth reports positive infinity. |
| M7–M8 | Strict crossing helpers distinguish touching equality; every getter/calculation preserves book state. |
| N1–N3 | Power-of-ten grids, exact decimal boundaries, one/three lots and valid large prices, quantities and order counts. |
| N4–N5 | Invalid price/size at different batch positions and sides, plus crossed/locked snapshots and BBOs, reject before mutation. |
| N6–N7 | NumPy lengths, shapes, dtypes, writable-buffer requirements and supported strided layouts. |
| N8–N9 | Reusing caller inputs cannot rewrite retained state; exported views/copies obey their documented reuse and lifetime contracts. |
| F1–F7 | Buffered snapshot startup, stale/future/equal-ID events, missing depth, bounded pending data, timeout/disconnect/cancellation cleanup and actual recorded replay. |

The standard book is unbounded. Advanced targets use a separately configured
bounded reference, which discards evicted depth at the end of each operation;
comparing only the top N of an unlimited reference would incorrectly resurrect
discarded levels later.

## Running the suite

Build both the library and native test modules first (`make build`). Run the
ordinary suite with all native modules required:

```sh
PYTHONPATH=src uv run pytest --require-native tests/orderbook tests/benchmarks/test_orderbook_collector.py
```

Ordinary CI includes ten fixed seeds with 2,000 generated actions per seed and
all three targets, rotating capacities 4, 7 and 64. Operation families are
deliberate, including BBO changes, duplicates, removals, capacity pressure and
reseed cycles. Additional depth calculations run at deterministic checkpoints.

The extended workload is explicitly deselected in ordinary runs. A dedicated
CI job runs forty seeds with 5,000 actions per seed through all three targets,
also covering capacity 1,024:

```sh
PYTHONPATH=src uv run pytest --require-native --orderbook-extended \
  tests/orderbook/test_scenarios.py::test_extended_orderbook_replay
```

The compact real Binance futures fixture and its provenance are in `data/`.
It is checked in and included in the sdist. Tests do not fetch live data.
Full-depth truth for aggregated exchange feeds is checked at completed depth
boundaries, while BBO events check their own quoted touch.

## Consolidated legacy coverage

The old cross-backend contract file and separate weak advanced property,
calculation and behavior collections are replaced by these shared cases.
Focused standard primitive/level, precision-trust, getter/iterator and
initialization contracts remain. Focused Python constructor/return-type and
meaningful numeric boundaries remain. Native C/engine tests remain for unsafe
memory and primitive behavior that a dictionary comparison cannot establish.
The C suite exercises both compiler-wide and portable weighted-mid arithmetic.
No production implementation is used as the reference for another.

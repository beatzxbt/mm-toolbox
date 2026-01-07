# Changelog

All notable changes to this project will be documented in this file.

## 1.0.0b0 (Unreleased)

### Notes
- Beta pre-release for the v1.0 feature set.

### Orderbook
- Always compute ticks/lots for advanced and standard orderbooks; removed optional input flags.
- Ignore deltas that would wipe the opposite side without replacement levels.
- Treat snapshots as full replacements for both ladders.
- Rename `update_bbo` to `consume_bbo` to align naming.
- Make C helper tick/lot conversions resilient to floating-point edge cases.
- Skip zero-lot insertions at the BBO to avoid phantom levels.
- Validate numpy ingestion arrays for length mismatches.
- Fix `OrderbookLevels.from_list` to default `norders` to 1 when omitted.
- Consolidate orderbook tests and expand edge-case coverage (crossing deltas, delete+insert same delta, capacity roll-right).

### Misc
- Validate `DataBoundsFilter` thresholds on initialization.
- Expose limiter `RateLimitState` in core and tighten Cython initialization paths.
- Add filter/limiter test coverage for configuration validation, thresholds, burst policy, and refill timing.

### Documentation
- Add a candles component overview and consolidate logging docs under `logging/`.

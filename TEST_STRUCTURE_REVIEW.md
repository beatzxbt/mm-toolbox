# Test structure review — 2026-10-25

Keep the coverage of past failures, but organize it by component and behavior.
“Regression” describes why a case was added; it does not need to describe its
permanent file or a separate test tier.

This review used a GPT-6 Sol agent with xhigh reasoning, as requested. No test
files have been moved or deleted. Publication is held for review.

## Current behavior

All these Python tests already use the normal pytest collection and the same
source and installed-wheel CI gates. There is no separate regression-test
runner. `--require-native` also requires all seven Cython test extensions.

The reviewed additions define 56 test functions across nine Python files;
parametrization produces additional collected cases. The maintenance issue
is the umbrella files crossing component ownership, duplicated helpers and
release-oriented names. Most scenarios add coverage missing from the older
happy-path tests.

## Recommended placement

| Current file | Destination and reasoning |
| --- | --- |
| `tests/candles/test_release_regressions.py` | Put invalid/stale input, replay, async delivery, cancellation and overrun scenarios beside the corresponding tests in `test_candles_base.py`; keep time limits in `test_candles_time.py`, volume splitting in `test_candles_volume.py`, and shared cross-type cases in `test_candles.py`. |
| `tests/websocket/test_release_regressions.py` | Put partial startup rollback in pool context-manager tests; lost connections and disabled recovery in pool error/connection-management tests; dead-fastest sends in pool send tests; terminal single-client close and cancellation in single-client tests; listener cleanup in connection/native tests. Reuse the existing WebSocket fixtures. |
| `tests/logging/test_logging_release_regressions.py` | Move standard drain/loss cases to standard handler tests, advanced multi-future shutdown cases to advanced handler tests, and keep shared remote-delivery semantics under a behavior-based contract name if they genuinely span both implementations. |
| `tests/rounding/test_rounding_power10.py` | Merge Decimal-oracle, adjacent-float and precision-limit cases into `test_rounding_rounder.py`, beside its existing boundary cases. Preserve scalar and vector coverage. |
| `tests/orderbook/test_reliability.py` | Retain the useful three-backend/two-orientation public behavior matrix, under a name such as `test_backend_contract.py`. Backend-specific tests do not replace last-write, zero-delete, transactional snapshot, crossing and VWAP contracts. |
| `tests/ringbuffer/test_ringbuffer_shm_reliability.py` | Keep shared SPSC/MPSC loss, destination-buffer, async and finalizer behavior together under `test_ringbuffer_shm_contract.py`; put producer leases and simultaneous claims in MPSC tests, and format rejection next to each mapping implementation. |
| `tests/ringbuffer/test_ringbuffer_shm_native.py` and its C harness | Retain a distinct native test. Its compile-time hooks force overwrite/read interleavings that Python cannot schedule. Preserve its source-path lookup, build flags and installed-wheel behavior when changing names. |
| `tests/test_release_artifact_validation.py` | Keep separate: this is an ordinary test of the release-validator tool, with its own component boundary. |
| `tests/orderbook/standard/test_binance_reference.py` | Keep the independent reference checked offline. Move shared replay types/helpers out of the live test module into non-test support code, so the offline test does not import another test module. |

## Preserve the missing scenarios

For example, the older WebSocket startup test checks that entirely failed
startup raises. The newer case checks rollback after one connection succeeds
while another is still pending. These are complementary. Likewise, candle
staleness before a close does not cover staleness after a reset, and a basic
duplicate-price test does not establish last-write/zero-delete behavior
across every orderbook backend.

Deduplicate only genuinely identical cases after mapping their inputs,
failure condition and assertions. Do not remove a stronger case just because
another test has a similar name.

## Strengthen while moving

- Replace the disabled-recovery test's fixed 0.25-second sleep with controlled
  scheduling/events and an assertion on replacement work.
- Replace tight logging shutdown wall-clock thresholds with controlled
  deadline checks while preserving cancellation, drain and loss assertions.
  The advanced handler's revised deadline test is an example of this.
- Extend extreme rounding-grid checks with adjacent representable values;
  checking only that a grid value rounds to itself is useful but limited.
- Update `tests/websocket/README.md`, whose file examples are stale.

## Suggested implementation order

1. Record the scenario inventory, collected parametrized cases and native
   requirements. The current baseline is 1,674 passed/eight skipped offline,
   or 1,677 passed/five skipped with the three live cases enabled.
2. Make mechanical moves and fixture consolidation in one reviewable commit,
   preserving assertions and parameter matrices. Keep production code out of
   that commit.
3. Make the deterministic timing and assertion improvements separately, so
   reviewers can distinguish organization from changed test behavior.
4. Run the normal source suite with required native modules and the installed
   wheel suite; verify the standalone SHM harness still compiles from both
   layouts. Run the release matrix before publishing the resulting revision.

This focused organization change is recommended before publication. A broad
rewrite of the test framework is not needed.

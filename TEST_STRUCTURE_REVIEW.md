# Test structure review — implementation status (2026-10-30)

## Integrated orderbook coverage

The approved orderbook follow-up replaces duplicated behavioral collections
with one independent dictionary reference and exactly three targets: the
standard Python book, advanced Python API and actual compiled Cython API.
The native adapter calls `cdef` methods directly. NumPy/scalar ingestion and
export ownership retain focused contract tests rather than complete extra
copies of the scenario matrix.

[The orderbook test guide](tests/orderbook/README.md) maps all agreed S, D, B,
C, M, N and F scenarios to their inputs and assertions. Ordinary CI includes
ten deterministic 2,000-action traces per target; a dedicated job runs forty
5,000-action traces, including capacity 1,024. Accepted operations compare the
whole retained book, metadata and BBO. Rejections and getters must preserve
state. Arithmetic expectations use independent integer/Decimal/Fraction logic.

The old `test_backend_contract.py`, advanced property, calculation, NumPy
ingestion and wrapper-boundary collections are removed after their stronger
replacement cases were added. Standard behavior, advanced integration,
boundary and small-capacity files now retain only their distinct API,
precision, constructor or limit contracts. Direct native C and engine tests
remain. There is one normal component collection, with the long workload
explicitly deselected unless requested; no named scenario silently skips.

Collector cases cover sequencing and failure cleanup beside the benchmark
driver. A checked-in compact real capture runs through the same three targets,
with independent raw-depth truth at completed depth boundaries. It is included
in the sdist so installed-wheel tests exercise the same replay.

Source and isolated installed-wheel suites passed **2,310 tests, eight skipped,
one extended case deselected** on Python 3.14 and 3.15 respectively, requiring
all seven native extensions. Cython-aware coverage is **86.53%**, above the
unchanged 85% gate. The standalone C suite
passed **40/40** through both compiler-wide and portable arithmetic, including
ASan/UBSan checks and the new large-unit conversion boundaries. The extended
run passed all **600,000 operation comparisons** in
**317.93 seconds**. Earlier counts and destinations below describe the previous
organization stage.

## Implementation status

The proposal below has been implemented in two mechanical test-organization
commits: `ae147b3` and `f04195f`. The moves retain the reviewed scenarios and
parameter matrices. One combined shared-memory legacy-layout check is now two
backend-specific checks, one each for SPSC and MPSC. The mechanical moves are
complete; publication remains held for review.

### Implemented destinations

| Area | Implemented organization |
| --- | --- |
| Candles | Cross-type limit validation is in `tests/candles/test_candles.py`; invalid-trade, replay, async delivery, cancellation, and overrun scenarios are in `tests/candles/test_candles_base.py`; quantity-conserving volume splits are in `tests/candles/test_candles_volume.py`. The shared trade factory is in `tests/candles/support.py`. The release umbrella was removed. |
| Rounding | Decimal-oracle, adjacent-float, precision-limit, and scalar/vector cases were folded into `tests/rounding/test_rounding_rounder.py`; `test_rounding_power10.py` was removed. Phase 2 adds six adjacent-representable-value cases at exponents -308, -100, -15, 15, 100, and 308. At 1e308, a request to round upward past the finite float range is expected to raise `ValueError`. |
| Orderbook | The initial three-backend/two-orientation contract move was superseded by the integrated reference suite described above. Live capture and offline reference tests still share `tests/orderbook/standard/binance_replay.py` support without importing the live test module. |
| WebSocket | Partial pool-start rollback, connection recovery, dead-fastest selection, single-client terminal-close/cancellation, and listener cleanup cases now live in the existing pool, single-client, and connection test modules, reusing shared fixtures in `tests/websocket/conftest.py`. |
| Logging | Standard drain/loss cases are in `tests/logging/standard/test_standard_handlers.py`; advanced future-shutdown coverage is in `tests/logging/advanced/test_advanced_handlers.py`; shared remote-delivery behavior is in `tests/logging/test_remote_delivery_contract.py`. |
| Shared-memory ring buffers | Shared SPSC/MPSC behavior is in `tests/ringbuffer/test_ringbuffer_shm_contract.py`; MPSC producer-lease and simultaneous-claim cases are in `tests/ringbuffer/test_ringbuffer_shm_mpsc.py`; legacy-format rejection is next to each mapping implementation. The combined legacy-layout case is now separate SPSC and MPSC cases, adding one collected case. |

### Collection and validation evidence

- The required-native baseline was 1,682 collected items. The focused candle,
  rounding, and orderbook inventory stayed at 692 through the mechanical moves;
  every old node ID has a recorded destination. Phase 2 adds six cases to that
  scope, bringing its current collected inventory to 698.
- The WebSocket/logging/ringbuffer inventory moved from 706 to 707 collected
  items. The recorded mapping explains the single-case increase from splitting
  the legacy-layout check.
- The organization/assertion work collects 1,689 cases: the 1,682 baseline
  plus the one SHM split and six phase-two rounding cases. The approved
  initial reliability work added 59 cases, collecting 1,748 at `eefd9a6`.
  Receive, subscription, TLS and heartbeat work adds another 74 cases,
  bringing the current collection to **1,822**.
- Mapping and count evidence is in `/tmp/mm-test-layout-analytics/` and
  `/tmp/mm-test-layout-io/`. The rounder phase-two case list and counts are in
  `/tmp/mm-test-layout-analytics/phase2-added-cases.txt` and
  `/tmp/mm-test-layout-analytics/phase2_counts.json`.
- The phase-two focused rounder run passed 63 tests; its log is
  `/tmp/mm-test-layout-analytics-phase2-rounding.log`. The root's phase-one
  focused component log is `/tmp/mm-test-layout-analytics-phase1-tests.log`.
- The final Python 3.14.4 source suite passed **1,814 tests, eight skipped**,
  with all seven native test extensions required. The standalone C orderbook
  suite passed **33/33**. The installed Python 3.15 wheel and Cython-aware
  coverage suites each passed the same **1,814/eight** inventory. Coverage is
  **85.75%** against the unchanged 85% gate. Detailed evidence and platform
  matrix status are linked from `EXAMPLES_BENCHMARK_REVIEW.md`.

### Retained native and tooling contracts

`--require-native` still requires the same seven Cython test extensions. The
standalone shared-memory native test and C harness remain separate from Python
backend contracts, with their source-path lookup, build flags, and installed-
wheel behavior preserved. `tests/test_release_artifact_validation.py` remains
an ordinary test of its own tool boundary.

### Assertion improvements and reliability coverage

The disabled-recovery check now synchronizes on a maintenance tick instead of
sleeping for 0.25 seconds. Standard and advanced logging close checks use
controlled deadlines while retaining drain, cancellation, loss and cleanup
assertions. Their focused run passed five tests; these changes did not alter
collection counts. The WebSocket README paths have been refreshed.

The reliability additions use the same ordinary pytest collection. Native
receive-loss lifetime checks sit in the connection suite; worker accounting
checks sit in advanced logging; example continuity and real subprocess
failure/cleanup checks sit in `tests/examples`; benchmark status/accounting
checks sit in `tests/benchmarks`. The installed-wheel runner now copies the
example and benchmark helper trees alongside tests, so those checks exercise
the installed library outside the source checkout. The sdist includes the
helper trees and native test sources. There is no separate regression runner.

The subsequent receive checks follow the same structure: burst/TLS behavior in
WebSocket integration tests, heartbeat ordering and native fields in connection
tests, subscription readiness in examples, and driver accounting/cleanup in
benchmarks. A deliberately small dedup history still forces full rollover, and
the startup-pressure fixture offers twice its configured receive capacity. This
preserves the failure boundaries without quadratic Cython tracing work or
larger timeouts. Default-capacity paced bursts remain covered independently.

## Historical proposal snapshot

This section preserves the review as it stood before implementation. Its
recommendations explain the organization choices above; “no files moved” and
the baseline counts below describe that review point, not the current tree.

Keep the coverage of past failures, but organize it by component and behavior.
“Regression” describes why a case was added; it does not need to describe its
permanent file or a separate test tier.

This review used a GPT-6 Sol agent with xhigh reasoning, as requested. No test
files have been moved or deleted. Publication is held for review.

## Baseline behavior at review time

All these Python tests already use the normal pytest collection and the same
source and installed-wheel CI gates. There is no separate regression-test
runner. `--require-native` also requires all seven Cython test extensions.

The reviewed additions define 56 test functions across nine Python files;
parametrization produces additional collected cases. The maintenance issue
is the umbrella files crossing component ownership, duplicated helpers and
release-oriented names. Most scenarios add coverage missing from the older
happy-path tests.

## Recommended placement (proposal)

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

## Suggested implementation order at review time

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

The review recommended this focused organization change before publication. It
has since been implemented as described above. A broad rewrite of the test
framework was not needed.

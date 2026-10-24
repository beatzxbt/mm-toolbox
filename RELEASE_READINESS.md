# Public release readiness and hardening

## Hardening status — 2026-10-19

The fixes are implemented on `fix/b9-release-hardening` for review against
`v1.0b9`. The supported contracts are lock-free overwrite-oldest SHM queues,
all-or-nothing pool startup, and power-of-ten rounding grids. No release has
been published.

The implementation covers the reported byte-buffer, SHM ownership/overwrite,
WebSocket lifecycle, candle delivery/replay, time EMA, rounding, orderbook,
limiter, filter, weights, logging and timestamp defects. CI now builds and
requires all seven native test extensions, validates installed wheels outside
the checkout and gates publishing on validation of the candidate revision.

### Local validation of the fixes

| Check | Result |
| --- | --- |
| Full suite with required native extensions | **1,657 passed, eight skipped** on Python 3.14.4 / macOS arm64. |
| Cython-aware coverage | **85.16%** aggregate (Python branches and Cython lines), passing the unchanged 85% threshold. Coverage uses compatible legacy Cython tracing; CI measures it on Python 3.14 while running native tests on 3.12–3.14. |
| Native C orderbook suite | **33/33 passed**, including ASan/UBSan. |
| Native SHM race/concurrency harness | Passed deterministic overwrite windows and consumed-plus-dropped accounting under normal, ASan/UBSan and TSan builds. |
| Installed-wheel suite outside checkout | **1,657 passed, eight skipped** in a fresh environment with current dependencies (including Cython 3.3.0, NumPy 2.5.4 and picows 2.3.1). Two nonfatal pytest fixture deprecation warnings remain. |
| Isolated distributions | Wheel built from sdist with current build dependencies; wheel and sdist pass twine check. Wheel contains 36 native modules, typing metadata and native SHM sources. |
| Ruff format/lint, ty, actionlint, diff whitespace | Passed. |

The skips are opt-in live-network cases. No native test module was silently
skipped. Tests with the locked development environment and a separate fresh
environment with current dependencies cover two dependency configurations;
they do not establish every minimum-version combination.

See [the SHM design and measurements](LOCK_FREE_SHM_DESIGN.md) for the memory
ordering argument and performance cost. The native ARM microbenchmark measured
about 52 ns for an eight-byte insert/peek/consume round trip. Large payloads
cost about 3.4–3.6 times the original unsafe core in this particular
uncontended driver. Production tail latency and cross-architecture behavior
still require measurement.

The remaining public release gates are listed below. The original findings
are retained as historical context for reviewing the changes.

## Python 3.15 and CI follow-up — 2026-10-24

- Added ordinary GIL-enabled CPython 3.15 support to package metadata and the validation/wheel matrices.
- Fixed the Linux native-test linkage failure by compiling the ladder and helper C sources into the test extensions that call them directly.
- Removed the generated C test executable from version control.
- Aligned picows build/runtime requirements at 2.3.1 and refreshed compiler/native dependencies; Cython 3.3.0 is now required.
- Full Python 3.15.1 suite: **1,657 passed, eight live-network tests skipped**. Additional release-artifact validation and moving-average tests: **139 passed** after fixture cleanup.
- Fresh Python 3.14.8 coverage build with the updated lock: **1,657 passed, eight skipped**, **85.17%** aggregate coverage.
- Local TLS 1.3 probes on Python 3.15 passed default certificate rejection, hostname verification, process-local test trust, all three WebSocket interfaces, and task/socket cleanup.
- Release checks require the version tag, workflow SHA, checked-out source and wheel/sdist metadata to agree.
- CI wheel jobs use native Linux x86-64/aarch64 and macOS arm64 runners, split by Python version; fresh remote validation remains pending.

## Initial review at 02d239f

Reviewed 2026-10-19 at `02d239f`, branch `perf/centralize-price-validation`, package version `1.0.0b9`.

**Recommendation: hold the production 1.0 release.** The feature set can remain frozen. There are reproducible correctness, delivery, and lifecycle defects to fix, and the automated release checks are incomplete. Passing the current suite does not establish absence of bugs.

The initial review used focused reproductions and validation before implementation began. Three Luna agents reviewed separate components; findings were checked against source, intended behavior, and reproduction output. Severity reflects production impact: P1 blocks release, P2 needs resolution or an explicit supported contract before release.

## Follow-up decisions

- SHM must remain lock-free. The synchronization-lock option is excluded. See [the implemented CAS/atomic-copy design and local measurements](LOCK_FREE_SHM_DESIGN.md).
- Pool startup is all-or-nothing: every requested connection must open before startup succeeds. Any failure closes successful opens and fails startup. Partial-startup behavior is excluded. Recovery after a successful start remains governed by the reconnect contract.
- The requested rounding domain is positive finite powers of ten, for example `0.001`, `0.01`, `0.1`, `1`, and `10`; values such as `0.2`, `0.3`, `0.5`, and `1.5` are excluded. This is a deliberate supported-domain restriction; exchanges do use other increments.
- Restricting the rounding domain does not make the current fast method correct unchanged. Additional focused checks found `bid(0.29) == 0.28` and `ask(0.07) == 0.08` at increment `0.01`, and `bid(1.0) == 0.99999` at increment `0.00001`. Preserve a fast precomputed-scale path, with correct on-grid boundary handling and explicit numeric limits.

## Progress and Git state

- The 1.0 beta changelog starts on January 10. Reachable history contains substantial implementation and hardening work in January, February, and especially May. May accounts for 99 commits including merges; July has six including merges. Commit counts describe activity, not engineering hours or productivity.
- The latest GitHub release is **b7, published May 12**. This HEAD is **23 commits ahead** of that tag, across **91 changed files, 6,518 insertions and 4,639 deletions**. Python 3.14 support, coverage work, substantial orderbook changes, and benchmark updates followed the release.
- The latest reachable commit is July 11: roughly 100 days before this review. Public release closure has been idle while b8/b9 remain marked unreleased.
- `origin/v1.0b9` is `df17aab`; the reviewed normalization change is one commit ahead, in open [PR #27](https://github.com/beatzxbt/mm-toolbox/pull/27).
- [PR #25](https://github.com/beatzxbt/mm-toolbox/pull/25) contains a useful build prerequisite for `test-all`, plus a changelog entry, and is absent from this HEAD. It does **not** fix the separate cleanup failure described below.
- [PR #24](https://github.com/beatzxbt/mm-toolbox/pull/24) targets the old b6 branch and has one unique commit while being 87 commits behind HEAD. Review its remaining logging changes deliberately; do not merge the old branch indiscriminately. [PR #18](https://github.com/beatzxbt/mm-toolbox/pull/18) is also an old release-line PR to reconcile administratively.
- The working tree was clean when review began. `master` still represents the older release line; choose and document the canonical 1.0 release branch before tagging.

The remaining effort is a hardening and release-validation phase. A readiness percentage or delivery estimate would be misleading until the P1 fixes pass their regression cases.

## Validation performed

| Check | Result and limits |
| --- | --- |
| Native library build | Passed with the existing Python 3.14.4 environment on macOS arm64. |
| Native test extensions | All six configured test extensions rebuilt before the final Python run. |
| Python suite | **1,437 passed, 8 skipped**, 77.35 seconds, with localhost socket access. Live exchange tests were not enabled. |
| Initial restricted run | Localhost bindings were denied by the sandbox. It also exposed an intermittent time-EMA reference failure under load; the unrestricted run passed. This is evidence of a timing-sensitive test/implementation combination, not 106 independent product failures. |
| C orderbook tests | **30/30 passed**. |
| C tests with ASan and UBSan | **30/30 passed**. This covers the standalone C orderbook suite, not every Cython extension or SHM operation. |
| Ruff lint / format / ty | Passed; 200 files already formatted. |
| Fresh local wheel and sdist | Built successfully using the installed build dependencies, without build isolation. Both passed `twine check`. |
| Wheel content / imports | 36 native modules, 32 stubs, 40 `.pxd` files; extracted wheel imported from `/tmp`, and a candle smoke check passed. This was not a fresh dependency environment or a full installed-wheel test suite. |
| Local WSS probe | A temporary TLS server exchanged payloads after its self-signed certificate was explicitly trusted; without that trust, verification correctly failed. Both tested picows transports worked. The local `BIO_get_init` import diagnostic was not fatal in this probe. This does not establish compatibility with all production TLS endpoints. |
| Full GitHub CI at HEAD | Failed on Python 3.12, 3.13, and 3.14 in coverage cleanup. Wheel jobs were skipped. Quick Check passed. |

GitHub's ordinary tests reported 1,257 passed / 14 skipped: six native test modules were skipped because CI builds only library extensions. Building them locally exercised 180 additional passing tests. The existing 85% coverage threshold has **not** been established for this candidate because CI stops before coverage runs.

The last automated b7 release run also failed its macOS wheel job and skipped publishing. Its logs have expired, so the historical root cause is unknown. This does not establish whether artifacts were uploaded manually. See [current CI run](https://github.com/beatzxbt/mm-toolbox/actions/runs/29149718961) and [b7 release run](https://github.com/beatzxbt/mm-toolbox/actions/runs/25742270014).

## P1: fix before the production release

### R1. Python-callable raw byte insertion can read outside its input or corrupt memory

[bytes.pyx:563](src/mm_toolbox/ringbuffer/bytes.pyx#L563) exposes `BytesRingBufferFast.insert_char` through `cpdef`. It checks only whether the supplied length exceeds the slot size, then passes it to `memcpy`. A negative length becomes a huge unsigned copy size. A positive length larger than the supplied Python bytes, but smaller than the slot, can read beyond that bytes object. The ordinary `BytesRingBuffer.insert_char` is also Python-callable at line 105 and accepts an independent pointer length.

Runtime introspection confirmed exposure in the freshly built wheel. The destructive negative-length case was not executed. The b7 changelog claims this Python exposure was removed, which the current source contradicts.

- [ ] Make pointer methods Cython-only, or provide Python wrappers that derive and validate the actual buffer length. Preserve the intended safe public API.
- [ ] Add isolated regression cases for negative, oversized, zero, and source-length-mismatched inputs; exercise the native path under sanitizers.

### R2. SHM overflow can overwrite an in-flight read and move the read cursor backwards

[shm_core.c:53](src/mm_toolbox/ringbuffer/shm/c/shm_core.c#L53) lets the producer update the consumer's cursor when dropping messages. [spsc.pyx:509](src/mm_toolbox/ringbuffer/shm/spsc.pyx#L509) checks that cursor before allocating/copying, but the producer can overwrite immediately after the check. The consumer later commits its stale cursor at [shm_core.c:176](src/mm_toolbox/ringbuffer/shm/c/shm_core.c#L176).

A deterministic C harness interleaved the existing functions at that gap: a full 64-byte ring, a consumer that peeked an 8-byte payload, then a producer insertion requiring three drops. The stale consume returned replacement bytes and changed `read_pos` from **48 back to 16**. This is an actual operation interleaving, not a probabilistic stress result. MPSC shards use the same writer.

- [ ] Coordinate overwrite with in-flight reads, or adopt a safe full-buffer policy. Merely changing the cursor store to CAS does not protect payload bytes during the copy.
- [ ] Verify monotonic cursors, valid payloads, exact drop accounting, and bounded behavior during concurrent overflow on Linux and arm64.

### R3. MPSC producer assignment reuses SPSC shards without exclusive ownership

[mpsc.pyx:319](src/mm_toolbox/ringbuffer/shm/mpsc.pyx#L319) assigns producer instances modulo the shard count. Each instance holds a separate cached write cursor; there is no shared write reservation or exclusive shard claim. Once instances collide, they can write the same position even sequentially.

A focused reproduction with two shards and four producer objects (the owner plus three attached producers) made two attached producers share a shard. Both inserts returned `True`; the consumer received only the second message. The master logger itself also consumes a producer assignment.

- [ ] Enforce exclusive shard ownership with explicit exhaustion/release behavior, or serialize/reserve writes safely across producers sharing a shard.
- [ ] Cover producer churn and more producer instances than shards, including the advanced logger's master and workers. An instance-local Python lock cannot coordinate separate processes.

### R4. A pool silently loses sends through its disconnected cached fastest socket

[pool.py:250](src/mm_toolbox/websocket/pool.py#L250) uses `_fast_conn` without verifying it is live. The outer check establishes only that some connection is live. [connection.pyx:234](src/mm_toolbox/websocket/connection.pyx#L234) silently ignores a send through the disconnected socket.

With two local sockets, closing the cached fastest left one healthy socket; `send_data()` returned normally and the server received no payload. The method explicitly documents sending latency-sensitive orders through this path.

- [ ] Invalidate/reselect on disconnect and validate the selected transport at send time. Make inability to send observable to the caller.
- [ ] Add a regression with one failed fastest connection and one healthy peer; verify exactly one delivery or an explicit error.

### R5. Pool startup and outage recovery can leave a dead pool marked connected

[pool.py:288](src/mm_toolbox/websocket/pool.py#L288) swallows connection exceptions. [pool.py:417](src/mm_toolbox/websocket/pool.py#L417) counts the resulting normal `None` returns as successes. A refused-handshake reproduction returned from `__aenter__` in `CONNECTED` state with zero active sockets. The eviction loop selects replacements only from connected sockets ([line 150](src/mm_toolbox/websocket/pool.py#L150)), so a complete outage leaves no replacements scheduled.

- [ ] Count actual successful connections; define startup failure explicitly and recover failed/missing connections according to the configured reconnect contract.
- [ ] Verify handshake refusal, partial startup, all-socket loss, and recovery after an outage. Also cover an initial exception in `new_with_reconnect`, which currently escapes the generator.

### R6. Closing a single WebSocket does not terminate reconnecting work or wake a waiting iterator

[single.py:129](src/mm_toolbox/websocket/single.py#L129) retains a reconnect iterator in the running `start()` loop. `close()` clears only the attribute and closes the current socket. A local reproduction observed a second accepted connection after close, with the client back in `CONNECTED` state. Separately, `__anext__` remained blocked after closing a quiet connection ([line 240](src/mm_toolbox/websocket/single.py#L240)).

- [ ] Give close a persistent stop signal, terminate owned reconnect/callback work, and wake pending consumers with a defined terminal result.
- [ ] Verify close during connection, backoff, active callbacks, and idle iteration, including repeated close and context-manager exit.

### R7. Candle iteration and replay can produce incorrect strategy inputs

- **Duplicate async delivery:** [base.pyx:213](src/mm_toolbox/candles/base.pyx#L213) resolves a waiting future and leaves the same candle as the next event. Two successive awaits returned the identical candle. The single event slot also retains only the latest candle when multiple close without a waiting consumer; define the intended queue/broadcast contract.
- **Stale trades after close:** staleness checks the current candle's timestamp, then resets it to zero ([base.pyx:191](src/mm_toolbox/candles/base.pyx#L191)). `TickCandles(1)` accepted times 2000 then 1000 and emitted the older candle second.
- **Historical time replay:** [time.pyx:31](src/mm_toolbox/candles/time.pyx#L31) starts its boundary from construction wall time. Replaying four historical trades ten seconds apart into ten-second candles closed zero candles and put all four trades in the current candle.
- **Nonterminating split:** [volume.pyx:53](src/mm_toolbox/candles/volume.pyx#L53) and [multi.pyx:68](src/mm_toolbox/candles/multi.pyx#L68) accept thresholds at or below a fixed epsilon, then repeatedly close empty candles without consuming input. Threshold `1e-13` with trade size `2e-12` timed out in bounded child-process reproductions for both classes. `TimeCandles` also lacks positive/finite interval validation.

- [ ] Resolve all four cases with persistent timestamp state, event-time replay boundaries, explicit async delivery semantics, and progress-guaranteed splitting.
- [ ] Assert chronological output, exactly-once delivery for the chosen consumer contract, and conservation of trade size/volume over bucket splits. Reject invalid/nonfinite configuration before processing.

### R8. Time EMA quantizes all elapsed time to whole wall-clock seconds

[tema.pyx:91](src/mm_toolbox/moving_average/tema.pyx#L91) uses integer `time_s()`. All updates within one second get alpha zero. A one-second-half-life EMA initialized at 10 returned 10 after updating to 100 in the same second. Wall-clock corrections can also make elapsed time negative. A reference test failed once under build load, then passed in the final suite, because it uses real one-second sleeps rather than controlled timestamps.

- [ ] Use a sufficiently precise monotonic clock for elapsed-time calculations and make the tests control elapsed time deterministically.
- [ ] Verify subsecond movement, elapsed-time recurrence, preview behavior, and reinitialization. Keep the existing API unless a separate contract decision is necessary.

### R9. Rounding can emit invalid exchange increments

[rounder.pyx:23](src/mm_toolbox/rounding/rounder.pyx#L23) treats every increment at least 1 as having integer decimal precision. With tick and lot sizes **1.5**, `ask(3.1)` and `size(3.1)` both returned **5.0**, which is off-grid; their configured ceiling is 4.5.

- [ ] Preserve tick/lot multiples for fractional increments above 1 as well as below 1.
- [ ] Check grid membership and directional bounds over representative increments and magnitudes, using an independent decimal/integer reference.

### R10. The release pipeline does not establish the advertised validation

[Makefile:51](Makefile#L51) attempts to delete nonempty `__pycache__` directories with `find ... -type d ... -delete`, including inside `.venv`. All three current CI test jobs fail here when the coverage step calls `remove-build-lib` ([ci.yml:49](.github/workflows/ci.yml#L49)). Subsequent coverage, quality, and wheel jobs do not complete.

CI builds only library extensions before `test-all`, so the six native test modules skip. The sibling PR #25 adds the missing build prerequisite. `cibuildwheel` tests also need a deliberate native-test strategy rather than accepting those skips. The sdist currently includes Python tests but none of their `.pyx`, C, or header sources.

- [ ] Fix scoped cache cleanup and build both library and native tests from a clean checkout. Make missing native test modules a failure in the intended CI jobs.
- [ ] Establish passing Cython-aware branch coverage at the configured threshold, then complete wheel validation on every supported platform/Python combination.
- [ ] Make publishing depend on successful validation of the exact release SHA. The release workflow currently has build dependencies but no full lint/type/coverage/test gate.
- [ ] Unify platform declarations: workflow environment variables expand architectures beyond the narrower `pyproject.toml` targets.

## P2: resolve correctness contracts and operational behavior

| Item | Evidence and required outcome |
| --- | --- |
| Duplicate orderbook ticks | Advanced snapshots and bulk deltas preserved duplicate price ticks in focused reproductions. See [core.pyx:160](src/mm_toolbox/orderbook/advanced/core.pyx#L160) and [orderbook_ladder.c:147](src/mm_toolbox/orderbook/advanced/c/orderbook_ladder.c#L147). Reject duplicates or normalize consistently; verify final quantities and total depth. The existing duplicate-snapshot test checks only a nonempty result. |
| Volume-weighted midpoint contract | [standard/orderbook.py:504](src/mm_toolbox/orderbook/standard/orderbook.py#L504) promises average execution prices, but both implementations select terminal level prices. For asks `100×1, 110×9`, bids `99×5, 80×5`, size 10, the documented VWAP midpoint is 99.25; the implementations return 95. Resolve the calculation/documentation contract before consumers use it for pricing. |
| Crossed-book and midpoint contracts | Standard ingestion permits crossed state while advanced ingestion has guards. Define supported malformed-feed behavior. Whole-tick flooring of midpoint is explicitly encoded in tests and was **not classified as a definite bug**; document it accurately instead of silently changing it. |
| Rate limiter overflow | [limiter.pyx:220](src/mm_toolbox/rate_limiter/limiter.pyx#L220): consuming one token, then requesting `2**63-1`, returned allowed and made usage negative. Guard arithmetic before addition and reject unsupported counts without mutating state. Extreme input, but confirmed. |
| Negative filter inputs | [misc/filter/core.pyx:27](src/mm_toolbox/misc/filter/core.pyx#L27): centering on -100 reverses upper/lower bounds; repeated unchanged -100 triggers updates. Support signed values or enforce/document a positive-only domain. |
| Zero-alpha normalized EMA weights | [weights/ema.py:30](src/mm_toolbox/weights/ema.py#L30): accepted `alpha=0` produced all NaNs. Reject it for normalized weights or define a finite limiting result. |
| Logging from async applications | Standard Discord/Telegram handlers call `asyncio.run()` during synchronous flushing. A focused async-loop repro failed delivery and emptied the logger batch. Define an async-safe supported delivery path. |
| Logging delivery and shutdown | Advanced remote handlers do not check unsuccessful HTTP status; 429/500 responses can look successful. Review concurrent rate-limit waiters and bound total shutdown time: [base.py:292](src/mm_toolbox/logging/advanced/handlers/base.py#L292) applies a timeout per future, and [master.pyx:353](src/mm_toolbox/logging/advanced/master.pyx#L353) joins without a timeout. Verify retries/error reporting and shutdown with a stalled handler. These paths were inspected, not exhaustively fault-injected. |
| File and interval logging semantics | Advanced `FileLogHandler(create=True)` truncates an existing log file, confirmed by repro; its implementation explicitly comments this behavior. Make the caller-facing contract unambiguous and suitable for restart. Standard interval flushing happens only when another log arrives; its README still describes a background thread. |
| Loss visibility and feed validity | Overwrite-oldest is an intentional documented buffer policy, **not itself an unexpected bug**. WebSocket and worker-logger callers do not receive adequate loss information. Surface saturation/drops and define the action required for sequence-dependent market data. Pool payload-hash deduplication does not establish exchange-sequence continuity or recover gaps. |
| Negative ISO timestamps | `time_iso8601(-0.5)` returned `1970-01-01T00:00:00.-500Z`. Low trading impact; normalize or reject unsupported pre-epoch input. |

## Remaining gates before a public production release

The implementation PR targets **v1.0b9** from **fix/b9-release-hardening**. It includes the reviewed normalization commit and the native-test build prerequisite. Earlier PRs remain open for their owner to reconcile; nothing has been merged or published.

1. Review the fixes and compatibility changes: SHM format 2 requires recreated mappings; each producer requires its own shard; crashed claims are not automatically stolen; rounding accepts powers of ten only; disconnected sends raise; closing WsSingle is terminal; oversized log records are explicitly dropped within the configured batch limit.
2. Obtain green CI and installed-wheel validation for the exact reviewed commit on Python 3.12–3.14, Linux x86-64/aarch64 and macOS arm64. Verify both minimum supported and current dependency versions before promising compatibility across that range.
3. Run a production-shaped soak covering saturation, producer churn/crashes, reconnect outages, half-open sockets, TLS, slow handlers and shutdown under load. Record bounded memory, drop/error counts, recovery times, throughput and tail latency. Atomic payload copying costs more for large SHM records; assess representative trading payloads against the actual latency budget.
4. Verify exchange-sequence gap detection and resnapshot behavior in the consuming integration. Payload-hash deduplication and overwrite diagnostics alone do not establish feed continuity.
5. Freeze the accepted SHA, reconcile the older release branches/PRs, update stable version/tag metadata when appropriate, rehearse TestPyPI installation and verify artifacts before a PyPI release.

No finite test run establishes complete absence of bugs. The PR is ready for code review once local checks pass; public production readiness still depends on the gates above.

## Evidence and review limits

Focused scripts and outputs are in `/tmp/mm-release-review-analytics.py`, `/tmp/mm-release-review-native.py`, and `/tmp/mm-release-review-streaming/`. The deterministic SHM harness is `/tmp/mm-release-shm-interleaving.c`. Local suite/build logs use `/tmp/mm-release-*`; fresh distributions are in `/tmp/mm-release-artifacts/`. These are temporary review artifacts.

The review did not exercise live exchanges, the full supported wheel matrix, all dependency versions, long-duration production load, or every native extension under sanitizers. It found no need to expand the feature set. The release decision should be based on closing the reproducible failures and obtaining the missing validation evidence for a fixed candidate.

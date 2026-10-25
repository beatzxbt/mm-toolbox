# Examples, benchmarks and release review — 2026-10-25

Publication remains on hold in [draft PR #31](https://github.com/beatzxbt/mm-toolbox/pull/31).
Two GPT-6 Luna max agents exercised the examples and benchmarks; a GPT-6 Sol
xhigh agent reviewed test organization. After approval, two GPT-6 Luna max
agents implemented that organization and assertion work, and a GPT-6.1 Sol
max agent implemented the remaining reliability fixes. The parent reviewed
the changes and ran the final integration checks.

All four examples and all 13 benchmark entry points have now executed real
work successfully with bounded settings during the initial audit. The data
collector also captured ten public messages without changing the tracked
100,000-line orderbook fixture. This establishes basic execution and cleanup;
the approved follow-up below adds explicit failure on unproven continuity.

The historical performance measurements below used a freshly built, installed `1.0.0b10` wheel from
`e4ec6c84958d1a4a6797005254e36558ed8724c7`, with locked runtime dependencies.
Those measurements predate the current reliability changes; they are retained
as audit evidence, not presented as new performance measurements of this PR's
final revision.

## Approved reliability and test follow-up

- WebSocket single/pool wrappers expose exact lifetime receive-overwrite
  totals, including startup, failed handshakes and replaced/retired sockets.
  Close preserves the totals. Native overwrite warnings are sampled at
  counts 1, 2, 4, 8, ... while exact counters remain available. The public
  runtime-checkable client protocol is unchanged; the concrete getters are
  additive. The receive path adds no locks.
- The examples stop on receive/SHM overwrite, rejected SHM writes, disconnects
  or sequence gaps, including loss before the first state update. They disable
  automatic reconnect, propagate candle/processor/child failure and clean up
  owned tasks, processes and files. Restart the whole example after failure;
  a book snapshot alone cannot repair missing candle or moving-average input.
- The Binance futures adapter follows the exchange's snapshot bridge
  (`U <= lastUpdateId <= u`) and subsequent `pu == previous u` rule. Buffered
  updates start before the snapshot fetch, and no book state is published
  before the initial bridge. MPSC readiness waits for every snapshot; its BBO
  streams update the touch of seed depth rather than maintaining full depth.
- Worker logging exposes cumulative rejected-message, transport-error and
  overwritten-batch counters after shutdown. The master reports loss that
  occurred before consumer attachment. The benchmark waits for shutdown
  before worker success, verifies worker identity/status/exit and fails on
  lifecycle errors or unexplained missing consumption. Intentional overwrite
  rows remain supported, with batch units made explicit.
- Test scenarios now live beside their components, with shared replay helpers,
  deterministic deadline checks and six additional extreme rounding-boundary
  cases. New continuity/accounting tests use the normal source and wheel
  gates. See [the completed test-structure review](TEST_STRUCTURE_REVIEW.md).
- Installed-wheel tests copy example/benchmark helpers outside the checkout;
  those helpers and native test sources are present in the sdist. Native
  library layouts changed, so rebuild the library and dependent Cython
  extensions together. The SHM core and overwrite-oldest policy are unchanged.

The adapter sequencing follows the [Binance futures local-book guide](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly).

### Final follow-up validation

The final source suite on Python 3.14.4 passed **1,740 tests, eight skipped**
with all seven native test modules required; the standalone C orderbook suite
passed **33/33**. The final collection has 1,748 cases: the prior 1,682, one
split backend check, six rounding boundary cases and 59 reliability cases.
The draft PR shows the cross-platform CI status.

| Check | Final local result |
| --- | --- |
| Python 3.14.4 source, clean native rebuild | **1,740 passed, eight skipped**, 72.59 s. |
| Python 3.15.1 installed wheel, isolated test tree | **1,740 passed, eight skipped**, 85.86 s; all seven native test modules rebuilt against installed headers. |
| Python 3.14 Cython-aware coverage | **85.55%**, passing the unchanged 85% threshold; **1,740 passed, eight skipped**. |
| Distribution checks | Isolated wheel and sdist builds, a separate wheel build from the sdist, twine and version validation passed. The wheel contains 36 native modules, typing metadata and native SHM sources; the sdist contains example/benchmark helpers and the native test harness. |
| Opt-in live integration tests | **Three passed** against the installed Python 3.15 wheel, including single/pool WebSockets and exact-reference orderbook replay. |
| Live examples against the final Python 3.15 wheel | Candles 6 s, averages 6 s, stream 8 s and MPSC 5 s exited 0 with no stderr. Candles/averages/stream processed live updates; the short MPSC run established all four snapshots and started workers. |
| Extended MPSC live check | The 12 s run processed live trades/BBO, then detected receive overwrites and exited **1**, reporting child failure. This is the intended stop-on-loss result; it demonstrates a workload limitation on this machine, not loss-free operation. Owned temporary files were removed. |
| Native receive-loss fault injection | A localhost burst into a one-slot test buffer makes the real multiprocess stream fail before state output and removes owned files. Included in both full suites. |
| Advanced logging final-status smoke | One worker, 32-byte messages, one 50 ms window: **144,976 attempted/consumed**, no rejected messages, transport errors, overwrites, worker errors, forced exits or drain timeouts. Exit 0; functional evidence rather than a new performance measurement. |
| Ruff lint/format, configured source typing, whitespace | Passed. |

The eight default skips are the three opt-in live tests and five legacy Python
checks that cannot mutate Cython fields; no required native module was skipped.
Workload processes, owned example paths and logging IPC files were absent
after the final runs. Evidence: `/tmp/mm-review-full314-final.log`,
`/tmp/mm-review-wheel315-tests.log`, `/tmp/mm-review-final-coverage314-tests.log`,
`/tmp/mm-review-final-dist315-build.log` and `/tmp/mm-review-final-live315/`.

## Environment and interpretation

Apple M2, 8 cores, 16 GiB RAM, macOS 26.6.2 arm64, ordinary GIL-enabled
CPython 3.15.1. Workloads ran serially without `PYTHONPATH=src`. Native runs
and analytics runs used separate quiet windows; these are local reference
measurements, not a cross-machine comparison or production latency guarantee.

Most per-call rows use 100,000 measurements after 1,000 warmup calls.
`numpy.percentile` uses its default linear interpolation. Python call wrappers
and timer overhead remain in the samples. Inverse mean latency is not treated
as end-to-end delivery throughput.

## Historical example measurements

| Example | Final bounded check | Result and limits |
| --- | --- | --- |
| `candle_types_demo.py` | 6 seconds, plus a 45-second instrumented capture | Exit 0, real trades/candles, clean cancellation. The filtered capture received 568 frames, skipped three zero-value inputs, and built 578 completed candles without zero lows/closes. |
| `moving_averages_demo.py` | 6 seconds | Exit 0, live BBO prices and all four averages, no stderr in the final run. Two connections exercise the pool. |
| `binance_stream.py` | 8 seconds, two child processes and three feeds | Exit 0, snapshot, trades, TEMA and candles; children stopped. One `WsSingle` per feed reduces redundant work. The latest run still logged **44 receive-buffer overwrites**. Treat this as observed loss requiring consumer recovery, not a loss-free feed. |
| `shm_mpsc_pipeline.py` | 5 seconds, four workers | Exit 0, all four snapshots and worker streams started, children and owned temporary files cleaned up; no stderr in the final run. Earlier runs did show receive loss. |

The candle anomaly was traced to real incoming trade-shaped JSON carrying
`p="0", q="0", X="NA", st=1`. An unfiltered 45-second capture received 725
frames, including three such inputs, and produced seven candles with zero
low/close. Direct positive-trade and five-builder async probes were correct.
The general `Trade`/candle API explicitly accepts zero values; the Binance
execution examples now skip and report nonpositive/nonfinite execution inputs.
No candle-core initialization change was made.

## Initial driver repairs

- Bring standard logging benchmark configuration, handler push and shutdown
  calls into line with the synchronous API; verify produced/consumed counts.
- Replace removed bytes/numeric ringbuffer operations with supported peeks and
  `unwrapped`, and label refill-plus-drain workloads accurately.
- Honor explicit zero warmup/measurement overrides in the shared runner.
  Preserve a replay suite's measured interval instead of overwriting it with
  JSON loading/warmup time.
- Make operation-count aliases consistent; restrict the rounding tick sweep
  to powers of ten. Give the batch-based candle benchmark a practical default
  of ten measured batches and two warmup batches (default run also passed).
- Count successful MPSC writes, rejected writes, overwritten records and
  remaining records; assert conservation after workers stop. Bound child
  waits and result collection, propagate failures, and clean up queues/maps.
- Use a counted local WebSocket sink for outbound send benchmarks, check every
  expected received frame, and distinguish send-call latency from RTT.
- Give live examples duration limits, awaited task cancellation, joined child
  shutdown, unique owned temporary paths and parent-visible child failures.
  Reserve the master's producer shard in the MPSC example.
- Fix logger/master startup ordering and avoid overwriting the checked-in
  market-data fixture with the collector's default output.
- Remove the advanced logging driver's wait for an unreachable delivery count
  after overwrite. Use bounded final drain, report lifecycle errors, and make
  its worker-active time interval and scenario-level percentiles explicit.

## Analytics benchmark results

Candle rows use 100 measured batches of 10,000 trades, five warmup batches,
1,000 retained candles. Values include factory setup and the batch loop;
percentiles are distributions of whole batches, not individual trades.

| Candle | No trade storage: mean ns/trade | No storage: trades/s | With storage: mean ns/trade |
| --- | ---: | ---: | ---: |
| Time | 254.80 | 3,924,652 | 260.29 |
| Tick | 256.40 | 3,900,111 | 282.34 |
| Volume | 296.00 | 3,378,402 | 430.27 |
| Price | 265.86 | 3,761,329 | 285.94 |
| Multi | 342.47 | 2,919,937 | 380.76 |

Moving averages use window 100 and 100,000 samples per operation. Normal and
fast modes both passed. Normal-mode mean update latency: EMA **292.2 ns**,
SMA **504.5 ns**, TEMA **360.1 ns**, WMA **502.6 ns**. Initialization rows also
include allocating a new instance; their means were 10.5–14.9 microseconds.

Rate-limiter default and state/burst/per-second configurations both passed.
With state/burst enabled, mean allowed/denied/burst latencies were
**194.6 / 207.5 / 226.2 ns**; p99 values were **291 / 292 / 334 ns**. Refill
setup is outside those individual call timers.

Rounder scalar and vector modes, all four power-of-ten tick sizes, and array
sizes 1/4/16/64 passed. At tick 0.01, lot 0.001 and 64-element arrays, scalar
bid/ask/size means were **215.4 / 214.0 / 218.1 ns**. Vector means were
**633.8 / 618.3 / 607.8 ns per array call**, not per element.

## Logging benchmark results

Standard logger: 128-byte messages, 100,000 measured emits, 1,000 warmup,
1,000 messages per timed emit batch, 200 shutdown repetitions. All **101,000
produced messages were consumed**, loss zero. Mean emit compute cost was
**1,151.7 ns/message**; p50/p95/p99 were **1,129.5 / 1,237.2 / 1,521.4 ns**.
Those percentiles describe repeated batch averages, not independent
per-message tail latency. Flushing 128 queued messages at shutdown averaged
**1.73 microseconds**, p99 **3.21 microseconds**.

Advanced logger: three repetitions of a one-second producer-active window,
flush interval 20 ms. Startup/drain are excluded from the rates below; the
rate is not a per-message latency measurement. Worker errors, forced exits
and drain timeouts were zero in every row. Only three scenario-level samples
exist per operation, so its printed percentiles cannot establish message
p99 delivery latency.

| Message bytes | Workers | Attempted | Consumed | Lost | Consumed / producer-active second | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 7,574,403 | 7,574,403 | 0 | 2,524,793 | 0% |
| 256, size sweep | 1 | 7,066,031 | 4,933,825 | 2,132,206 | 1,644,523 | 30.18% |
| 256, worker sweep | 1 | 6,670,336 | 4,768,112 | 1,902,224 | 1,589,366 | 28.52% |
| 256, worker sweep | 2 | 15,264,240 | 8,432,199 | 6,832,041 | 2,810,712 | 44.76% |

These saturated rows intentionally exercise overwrite-oldest transport.
Accepted/attempted throughput must not be presented as loss-free delivery.

## Historical benchmark validation and evidence

All 17 canonical native/transport smoke/timed commands and all 11 final
analytics commands exited 0. The final four example commands and the default
candle benchmark exited 0. Ruff lint/format, configured source type checking
and diff whitespace checks pass. These were driver-only repairs at the audit
point. The approved follow-up above changes native library layouts and
requires a clean rebuild.

Raw local evidence and exact commands:

- Native/transport: `/tmp/mm-toolbox-beta-owned-b10/{RESULTS.md,summary.json,commands.txt}`.
- Analytics: `/tmp/mm-beta-root-analytics/runs.json` and matching stdout/stderr.
- Final examples: `/tmp/mm-beta-root-final-examples/runs.json`; the latest
  single-feed stream output is `binance_stream_single_feed.stdout/.stderr`.
- Live input diagnosis: `/tmp/mm-beta-b10-live-candle-capture-45.json` and
  `/tmp/mm-beta-b10-live-candle-capture-filtered.json`.
- Initial failures: `/tmp/mm-beta-examples-audit-luna6/`.

The previous generic `/tmp/binance_logger` artifact was verified unused and
removed. Final child processes and owned temporary paths were checked.

## Way forward

Keep publication held until the review is accepted. The focused test
organization and explicit stop-on-lost-continuity policy are now implemented.
Review the changes and final validation in the draft PR before choosing a new
release revision. Existing candidate tags remain unchanged and unpublished.

The detailed native/transport measurements follow.

# Native and transport benchmark details

## Environment and method

Python 3.15.1, `mm-toolbox` 1.0.0b10 installed wheel, macOS 26.6.2 arm64, Apple M2 (8 cores, 16 GiB). Timed commands ran serially with the wheel environment and no `PYTHONPATH=src`. Every canonical smoke and timed command exited 0. Full commands, combined stdout/stderr, and status files are indexed in `summary.json` and `commands.txt`.

Latency percentiles use `numpy.percentile` with its default linear interpolation. Orderbook per-level latency divides each operation latency by the number of levels in that input message. Ringbuffer batch rows count batch calls; at the timed 100,000-operation setting, batch sizes 1/10/100/1,000 divide evenly, so each batch setting processed exactly 100,000 items. For non-divisible inputs, the benchmark uses `ceil(N / batch_size)` calls, so report the actual count as calls × batch size.

The ringbuffer `fill+consume_all` samples include `insert_batch` refill inside the timed callable, followed by `consume_all`; they are not pure `consume_all` latency. Numeric/bytes suite-wide “Total time” and inverse-mean ops/s combine different operations and are not end-to-end throughput.

## Orderbooks

The checked-in 100,000-line fixture was read without modification. Each run skipped the first snapshot from timing, warmed up on 1,000 messages, then measured the remaining 98,999 messages (90,019 BBO, 8,980 delta). The elapsed interval covers only measured message processing.

| Implementation | Levels | Hint | Measured time | Message throughput | BBO ns/level p50/p95/p99 | Delta ns/level p50/p95/p99 |
|---|---:|---|---:|---:|---:|---:|
| Standard | — | — | 1.189 s | 83,255 msg/s | 541.5 / 729 / 1,750 | 756.3 / 1,044.5 / 3,019 |
| Advanced | 2,048 | on | 0.299 s | 330,965 msg/s | 62.5 / 62.5 / 83.5 | 308.8 / 701.4 / 953.6 |
| Advanced | 4,096 | on | 0.483 s | 204,970 msg/s | 62.5 / 83.5 / 166.5 | 534.9 / 1,331.9 / 2,051.7 |

## In-process ringbuffers

Numeric ran 100,000 operations with 1,000 warmup at capacities 16, 64, 256, 1,024, and 4,096. Mean latency in ns (multi-size comparative output):

| Capacity | Insert | Consume | Contains | Peek right | Unwrapped |
|---:|---:|---:|---:|---:|---:|
| 16 | 282.3 | 1,377.9 | 492.0 | 71.9 | 263.3 |
| 64 | 272.0 | 1,489.0 | 530.4 | 70.2 | 299.0 |
| 256 | 289.6 | 1,468.2 | 566.8 | 70.7 | 276.2 |
| 1,024 | 285.3 | 1,341.5 | 951.8 | 68.5 | 264.6 |
| 4,096 | 270.2 | 1,295.1 | 511.5 | 75.1 | 260.3 |

At capacity 1,024, numeric p50/p95/p99 ns were: insert 250/292/375; consume 1,333/1,750/2,000; combined fill+consume_all 28,625/35,458/51,583; contains 500/583/709; peek left 83/84/125; peek right 83/84/125; unwrapped 250/292/375. The batch-call counts were 100,000 / 10,000 / 1,000 / 100 for batch sizes 1 / 10 / 100 / 1,000.

Bytes used capacity 256, item sizes 32/128/512/2,048 bytes, and 100,000 operations with 1,000 warmup. p50/p95/p99 ns:

| Item bytes | Insert | Consume | Combined fill+consume_all |
|---:|---:|---:|---:|
| 32 | 167/209/291 | 291/417/709 | 2,125/2,916/4,791 |
| 128 | 167/209/2,667 | 250/292/375 | 2,083/2,209/3,000 |
| 512 | 208/333/4,666 | 292/417/500 | 2,500/3,417/4,916 |
| 2,048 | 250/3,625/5,458 | 333/458/833 | 5,125/5,625/7,250 |

## Shared-memory ringbuffers

SPSC used a 1 MiB requested capacity, 100,000 insert/consume latency samples, and a separate 100,000-message producer/consumer run for each payload. The benchmark enlarged the mapped ring to 4 MiB for 32-byte payloads and 64 MiB for 512-byte payloads to fit all records without loss. Producer and consumer counts both matched 100,000. Latency p50/p95/p99 ns: for 32-byte messages, insert 125/166/208 and consume 333/666/791; for 512-byte messages, insert 125/208/1,208 and consume 375/500/1,375. The per-process throughput run was a single sample: 32-byte producer/consumer 5.85/5.88M msg/s; 512-byte producer/consumer 3.14/3.16M msg/s. These are producer/consumer process rates, not a separate end-to-end latency percentile.

MPSC used 1 MiB total capacity split across four 256 KiB sub-rings, one-second producer/consumer windows, payloads 128/512 bytes, producer counts 1/2/4, plus scalability and fairness runs. The reported counts below are for each nominal one-second consumer window; producer success includes accepted records that may overwrite older records. Accounting verifies successful inserts = consumed + overwritten + remaining, with zero rejected writes in these runs.

| Payload | Producers | Successful inserts | Overwritten | Consumer received | Remaining |
|---:|---:|---:|---:|---:|---:|
| 128 B | 1 | 4,006,409 | 18,856 | 3,987,553 | 0 |
| 128 B | 2 | 8,625,408 | 0 | 8,625,408 | 0 |
| 128 B | 4 | 13,789,599 | 8,423,329 | 5,363,861 | 2,409 |
| 512 B | 1 | 3,508,455 | 1,371 | 3,507,084 | 0 |
| 512 B | 2 | 8,206,729 | 1,094,448 | 7,112,281 | 0 |
| 512 B | 4 | 14,726,492 | 10,601,380 | 4,125,112 | 0 |

MPSC insert and consume latency samples were capacity-limited to 2,730 for 128-byte payloads and 1,365 for 512-byte payloads, below the requested 4,096. The timed MPSC output contains one sample for each throughput run, so those rows do not provide throughput percentiles.

## Local WebSocket benchmark

Configuration: 128-byte payload, 100,000 send calls per mode, 1,000 warmup send calls per mode, three pool connections. The local sink received the exact expected totals: 201,000 single-connection messages and 804,000 pool messages. Connect p50/p95/p99 was 0.409/0.656/0.926 ms for WsSingle and 0.734/1.153/1.894 ms for WsPool.

| Mode | Send-call duration | Calls | Frames | Measured rate | Send-call p50/p95/p99 |
|---|---:|---:|---:|---:|---:|
| WsSingle | 28.898 ms | 100,000 | 100,000 | 3.46M frames/s | 250/334/1,125 ns |
| WsPool fastest | 63.594 ms | 100,000 | 100,000 | 1.57M frames/s | 625/709/1,584 ns |
| WsPool broadcast | 138.856 ms | 100,000 | 300,000 | 2.16M frames/s; 0.72M calls/s | 1,167/1,750/2,500 ns |

The throughput timer covers synchronous `send_data()` calls. The sink confirms delivery after each timed section, but the send-call percentiles are not network RTT. The 100k runs logged transport high-watermark pause/resume notices; all expected frames arrived and both child processes exited 0.

## Cleanup and caveats

The collector captured 10 public Binance market-data messages to a `/tmp` JSONL file. Its default output now points to `btcusdt_capture.jsonl`, not the tracked `btcusdt_100k.jsonl` fixture. No SHM files or benchmark/collector child processes remained after the final runs. Results are single serial runs on this machine; repeated-run confidence intervals were not collected. Treat latency percentiles as observed sample distributions, and throughput samples as one-run results.

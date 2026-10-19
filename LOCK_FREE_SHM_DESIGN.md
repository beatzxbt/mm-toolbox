# Lock-free repair for SHM overwrite

Status: implemented on the release-hardening branch as SHM format 2. Deterministic overwrite interleavings, pthread stress, and ASan/UBSan/TSan pass locally on macOS arm64. This is a repository-specific design; the sources below support its primitives, not a formal verification of the whole algorithm. Cross-architecture validation and production-shaped latency/soak measurements remain release gates.

## Required behavior

- Keep a bounded, variable-length byte queue with overwrite-oldest behavior.
- Keep exactly one active producer per SPSC ring, including each MPSC shard.
- Do not wait for a reader to finish copying before a producer can overwrite.
- Return a record only if the consumer atomically wins retirement of that record.
- Use target-verified lock-free atomics; do not allow a library-backed lock fallback.

The nonblocking core can be lock-free even though existing blocking `consume()` wrappers wait when the queue is empty. Lock-free does not promise a fixed upper bound for an individual consumer under continuous overwrite.

## Storage and publication

Align record starts and spans to eight bytes: one atomic 64-bit length word followed by a payload rounded up to whole atomic 64-bit words. Zero-pad the final word from a local temporary; never read past the caller's input. Ring capacity and mapping offsets must preserve alignment.

All concurrent accesses to shared record storage must use atomic operations of the same width. Ordinary `memcpy` is permitted between private buffers, but not for concurrently accessed ring payloads. Update every path: insert/batch, consume/into/all/async, peeks, snapshots, and any Cython bypass helpers.

Producer writes complete data before release-publishing `write_pos`. Consumer acquire-loads publication before treating a record as available. Validate tentative length/span against capacity and the published range before allocating or copying. Reject/retry inconsistent snapshots without trusting an unvalidated length.

Padding adds up to seven bytes per record and requires a SHM format/version change. Old mappings must fail attachment validation, rather than mix two layouts.

## Consumption and overwrite share one retirement cursor

Replace unconditional `read_pos` stores with compare-and-swap (CAS).

Consumer attempt:

1. Load retirement cursor `r` and a consistent published write boundary.
2. Atomically read and bounds-check the tentative record header.
3. Atomically copy the payload into private storage.
4. Execute the required acquire ordering after all speculative shared reads.
5. CAS retirement cursor from `r` to the record's padded end.
6. Return the private copy only on CAS success. Otherwise discard it and retry.

Producer when full:

1. Load the retirement cursor and determine complete records to drop.
2. CAS that exact cursor to the end of those records.
3. If CAS fails, recompute using the updated cursor; do not apply the old drop count.
4. Only after winning retirement, overwrite reclaimed bytes using atomic stores.
5. Publish the completed new record normally.

A failed CAS means another participant made progress. Neither participant owns a lock or a busy flag that the other must wait to clear. This is a system-wide progress argument, not a wait-free per-operation bound.

## Why speculative copying can be safe

There are two relevant outcomes:

- Consumer wins retirement first: its private copy is complete before producer reuse. A producer's stale CAS fails and it must recalculate.
- Producer wins retirement first: a consumer may read replacement or mixed words, but those reads are atomic and bounded. Its retirement CAS must fail, so it never exposes that speculative copy.

The second outcome depends on memory ordering, not just CAS. A candidate ordering uses release stores for shared header/payload words, relaxed speculative word loads, an acquire fence after those loads, and acquire-release retirement CAS. If any speculative load observes an overwriting release store, the acquire fence establishes ordering from the producer's preceding retirement to the consumer's following CAS. That CAS therefore cannot successfully retire the old cursor. Publication separately orders reads of the original committed record.

This argument assumes no retirement-cursor ABA. Define supported counter lifetime/wrap handling explicitly. It also assumes exclusive producer ownership and no concurrent close/unmap of an active operation. These are required invariants, not conditions repaired by fences.

Use a stronger-order reference implementation when investigating interleavings; review the optimized orders separately. Do not substitute an ordinary copy on either side based only on observing that a CPU's loads are usually indivisible.

## Implemented integration details

- The implementation uses CAS winners to distinguish actual consumption from actual drops. The existing independent `msg_count` updates are not an atomic snapshot of publication/retirement; the implementation scans validated headers instead and never uses a count to establish availability.
- Reusable consumer scratch storage is used for `consume_into` so failed speculative attempts leave caller storage untouched without per-message allocation after growth.
- Peeks and `unwrapped()` use atomic copying plus final snapshot validation without retiring records.
- Batches reserve/drop safely, publish complete records, and account only for cursor advances they won.
- Headers separate producer and consumer cursor cache lines. A compile-time assertion requires always-lock-free aligned 64-bit atomic operations.
- A producer crash after dropping and before publishing can lose retired messages, but must not expose a partial message or leave a busy flag blocking the reader. Restart/recovery requires explicit exclusive writer ownership and a consistent published position.

Measure uncontended operations, overflow, forced consumer pauses, typical and maximum payloads, batch sizes, retry counts, and tail latency. Atomic word copying may lose vectorized-copy throughput; lock-free alone is not evidence of meeting the latency budget.

If that cost is unacceptable, investigate transferring ownership of whole pages to the reader so it can use ordinary `memcpy` on exclusively owned storage. Linux tracing uses a related reader-page exchange design, but its kernel assumptions, cross-page records, and publication behavior require a separate user-space design here.

## Sources supporting the primitives and limitations

- [WG21 P1478R8](https://www.open-std.org/JTC1/SC22/WG21/docs/papers/2022/p1478r8.html): retry checks do not make racing ordinary copies valid; speculative shared accesses must be atomic.
- [C++ draft: fences](https://eel.is/c++draft/atomics.fences): release-store/acquire-fence synchronization underlying the candidate copy validation ordering.
- [GCC atomic builtins](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html): atomic operations, memory orders, and checking whether the implementation is always lock-free.
- [Linux lockless ring buffer design](https://docs.kernel.org/trace/ring-buffer-design.html): reader-page ownership exchange as an alternative architecture, not a drop-in implementation for this queue.

## Local performance evidence

On macOS arm64 / CPython 3.14.4, an optimized standalone C driver performed 20,000 sequential insert/peek/consume round trips per payload, alternating the original reviewed core at 02d239f and the repaired core over seven runs. Median run means (nanoseconds per round trip):

| Payload bytes | Original unsafe core | Format 2 core |
| ---: | ---: | ---: |
| 8 | 51.2 | 52.1 |
| 64 | 52.0 | 55.3 |
| 256 | 57.1 | 70.2 |
| 4,096 | 153.3 | 515.6 |
| 65,528 | 2,141.7 | 7,679.5 |

These are uncontended native microbenchmarks on one machine, not Python API or tail-latency guarantees. Large payloads cost about 3.4–3.6 times as much here because shared copying uses atomic words instead of vectorized memcpy. Benchmark representative production payloads, contention, overload and tail latency before accepting this implementation for a specific trading latency budget. The native regression harness also provides a benchmark mode for repeatable measurements of the repaired core.

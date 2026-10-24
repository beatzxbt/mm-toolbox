# Shared memory queue format 2

SPSC queues and every MPSC shard have one exclusively claimed producer.
Creating an MPSC queue also claims a shard: provision `num_rings` for the
creator plus every simultaneously attached producer. Attachment raises
`RuntimeError` when all shards are claimed. `close()` and normal finalization
release a claim while preserving queued records. A process must attach a new
producer after `fork`; inherited objects cannot write or release the parent's
claim. Claims left by a crashed process are not stolen automatically. Recreate
the mapping after its users have stopped to recover that capacity.

Each record contains an eight-byte length followed by the payload padded to
an eight-byte boundary. All shared record accesses use aligned, native,
lock-free 64-bit atomic operations. A compile-time check rejects platforms
without those operations. Ring and sub-ring headers occupy 128 bytes; the
MPSC global header occupies 64 bytes. Format 2 magic values reject older
mappings. Capacities below eight bytes are rounded up to eight.

The producer makes space by CAS-advancing the retirement cursor past complete
oldest records. The consumer copies into private storage, then CAS-advances
that same cursor. Only a consumer that wins its CAS exposes the copy. No
producer waits for a consumer to finish, and no mutex, spinlock or reader busy
flag coordinates the native operations. Blocking wrappers wait for data when
empty. A consumer may retry under continuous overwrite.

Atomic payload stores use release ordering; speculative loads use relaxed
ordering. An acquire fence follows speculative reads before retirement or
snapshot validation. If any load observes replacement storage, its producer's
release store synchronizes with that fence. The producer's preceding cursor
retirement then happens before the consumer's cursor CAS/load. Atomic
coherence prevents a successful stale CAS or a valid stale snapshot. Original
records are ordered by acquire-loading the published write boundary. No
ordinary shared-memory copy is used on either side.

Positions are monotonic unsigned 64-bit byte offsets and never wrap. Inserts
fail before an offset addition would overflow; recreate the mapping after
stopping its users if this lifetime limit is reached. A producer crash after
retiring old records and before publishing a replacement can lose those
retired records, but cannot publish a partial replacement or block a reader.
Closing/unmapping concurrently with an active operation is unsupported.

`insert()` and `insert_batch()` retain their boolean success result. Inspect
`last_insert_dropped` immediately afterward for that operation's overwritten
record count. `dropped_count` is the shared cumulative count since mapping
creation (summed across MPSC shards), and `failed_inserts` counts rejected
operations for that producer instance. Only successful producer retirement
CAS operations increment drop counts. Timestamp properties remain monotonic
clock readings.

There is no separately incremented/decremented message count. `len()` scans
validated record headers within an acquired publication boundary and checks
the retirement cursor afterward. It cannot underflow and never establishes
whether a consuming operation succeeds. Counts and snapshots are coherent
within one shard; combining independently active shards is not a simultaneous
global snapshot. `consume_into()` and `consume_all_into()` use reusable private
scratch that grows only for a new largest payload. A failed speculative
retirement leaves caller storage untouched, with no per-message Python
allocation after the scratch has reached the required capacity.

Native regressions are in `tests/ringbuffer/shm_native_regression.c`. The test
hook forces overwrite between atomic payload word loads, and pthread stress
checks payload integrity, monotonic cursors and consumed-plus-dropped
accounting. Its `benchmark` mode reports native round-trip timings for 8, 64,
256, 4096 and 65528-byte payloads. Performance and weak-memory validation must
be checked on each supported architecture.

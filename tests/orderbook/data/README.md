`binance_futures_replay.jsonl` is an unmodified event prefix from a public
BTCUSDT Binance USD-M futures capture made by this repository's collector.
Only JSON whitespace was compacted. It contains no credentials or account data.

The REST snapshot has five levels per side. The two public WebSocket streams
are `bookTicker` and `depth@100ms`; the capture retains their original arrival
order, source update IDs, exchange timestamps, and receive timestamps. Exchange
`E` timestamps span 2026-10-27 01:03:32.812–01:03:36.175 UTC. The original file
was `mm-collector-verified-qa19gorw/capture.jsonl` (10,000 records), SHA-256
`64d423389de2e0c16b836bb7496b27c187041e204081f1d19f2e65ca86924abf`.

The fixture keeps source rows 0–949 inclusive: one snapshot, 33 continuous depth
batches, and 916 BBOs. Row 357 is the first captured BBO jump across multiple
retained price levels; row 949 is the next covering depth batch. Ending at that
depth boundary preserves the complete bootstrap and depth chain. The sequenced
publication gate emits 744 records (33 depth batches and 710 BBOs after the
snapshot), leaving 198 unverified future BBOs unpublished.

Tests compare every retained price/quantity/order field against an independent
dictionary. At depth boundaries they also compare against depth-only replay
that never consumes BBOs. This is the complete book known from the shallow
snapshot and captured updates, rather than a claim to reconstruct unobserved
exchange depth.

Fixture SHA-256:
`6291981b21f3cc953e5cd8a13e82da814dcee723c7c284e29e409b1a55b88e11`.

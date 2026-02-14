"""Shared low-level parser helpers.

The hot-path helper implementations live in `utils.pxd` as `cdef inline`
functions so Cython can inline them into parser loops.

Available helpers:
- `parse_u64_until_char`: parse ASCII digits into `unsigned long long` until a
  delimiter and leave the cursor on that delimiter.
- `parse_quoted_span`: return start/end pointers for a quoted string value and
  leave the cursor on the closing quote.
- `is_market_token`: specialized fixed-token check for `"MARKET"` used by the
  trade parser hot-path.
"""

# Parser Experiments

This folder contains narrow, high-performance parser experiments that are not
part of the public `mm_toolbox` API.

## Binance bookTicker parser

`binance_bookticker_parser.pyx` is a strict parser for Binance futures
`@bookTicker` payloads using the canonical key layout:

`{"e":"bookTicker","u":...,"s":"...","b":"...","B":"...","a":"...","A":"...","T":...,"E":...}`

Hot-path assumptions:
- payload shape/order is canonical and stable
- field names and field types are always valid
- payload has no whitespace/format deviations

This parser intentionally minimizes defensive checks for speed. If an anomaly
does occur, callers should catch exceptions and fall back to strict decoding
(`msgspec`) outside this module.

Performance note:
- `BinanceBookTickerStrictParser.parse()` returns a reusable mutable
  `BinanceBookTickerView` object (same object each call).
- `s` is interned/cached and `b/B/a/A` are parsed to Python `float` on parse.
- `E` and `T` are cached Python `int` objects when values repeat.
- `u` is stored as a C integer internally and exposed as Python `int` via the
  view property (materialized on access).
- For streaming workloads, instantiate once and reuse:

```python
from mm_toolbox.misc.parser.binance_bookticker_parser import BinanceBookTickerStrictParser

parser = BinanceBookTickerStrictParser()
parse = parser.parse
parsed = parse(payload)
# parsed is a mutable view object reused across calls.
# Read fields immediately if you need immutable snapshots.
```

## Binance trade parser

`binance_trade_parser.pyx` is a strict parser for Binance futures `@trade`
payloads using the canonical key layout:

`{"e":"trade","E":...,"T":...,"s":"...","t":...,"p":"...","q":"...","X":"...","m":...}`

Performance note:
- `BinanceTradeStrictParser.parse()` returns a reusable mutable
  `BinanceTradeView` object (same object each call).
- `s` and `X` are cached/interned.
- `X="MARKET"` uses a dedicated hot-path.
- `p` and `q` are parsed to Python `float` on parse with recent-value caches.
- `E`, `T`, and `t` are stored as C integers internally and exposed as Python
  `int` via view properties (materialized on access).

## Build

```bash
uv run python3 - <<'PY'
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        name="mm_toolbox.misc.parser.binance_bookticker_parser",
        sources=["src/mm_toolbox/misc/parser/binance_bookticker_parser.pyx"],
        extra_compile_args=["-O3", "-Wno-sign-compare", "-Wno-unused-function", "-Wno-unreachable-code"],
    ),
    Extension(
        name="mm_toolbox.misc.parser.binance_trade_parser",
        sources=["src/mm_toolbox/misc/parser/binance_trade_parser.pyx"],
        extra_compile_args=["-O3", "-Wno-sign-compare", "-Wno-unused-function", "-Wno-unreachable-code"],
    ),
]

setup(
    name="misc-parsers-build",
    package_dir={"": "src"},
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False, "cdivision": True},
    ),
    script_args=["build_ext", "--inplace"],
)
PY
```

## Shared Utils

`utils.pxd` contains shared `cdef inline` helpers used by both parser modules:
- unsigned integer parsing up to a delimiter
- quoted-span boundary parsing
- `X="MARKET"` token check

`utils.pyx` exists as the module companion file and keeps implementation details
co-located with the parser experiments.

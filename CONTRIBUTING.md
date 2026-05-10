# Contributing to MM Toolbox

Thank you for contributing to MM Toolbox. This document covers everything you need to know: environment setup, code standards, architecture patterns, testing requirements, and the release workflow. Read it in full before opening your first PR.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Development Workflow](#development-workflow)
4. [Architecture & Design Patterns](#architecture--design-patterns)
5. [Code Standards](#code-standards)
6. [Documentation Standards](#documentation-standards)
7. [Testing Standards](#testing-standards)
8. [Build System](#build-system)
9. [Commit Guidelines](#commit-guidelines)
10. [Pull Request Process](#pull-request-process)
11. [Release Process](#release-process)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python**: 3.12 or 3.13 (CPython only)
- **uv**: Modern Python package manager ([install guide](https://docs.astral.sh/uv/))
- **C compiler**: GCC (for Cython/C extensions)
- **Make**: For build automation

Verify your environment:

```bash
python --version  # Should be 3.12+ or 3.13+
uv --version      # Should be 0.5+
make --version    # Should be GNU Make 3.81+ or equivalent
```

---

## Environment Setup

```bash
# 1. Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/mm-toolbox.git
cd mm-toolbox

# 2. Install all dependencies (including dev, build, and test groups)
uv sync --all-groups

# 3. Build Cython extensions for the first time
make build-lib

# 4. Build test extensions (needed for C/Cython tests)
make build-test

# 5. Verify everything works
make test-all    # Run C + Python test suites
make fix         # Format + typecheck
```

**What `uv sync --all-groups` installs:**

- Runtime dependencies (numpy, picows, ciso8601, msgspec, aiohttp, xxhash)
- Dev tools (pytest, ruff, ty for type checking)
- Build tools (Cython, setuptools, wheel)

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming convention:

- `feature/description` — New functionality
- `fix/description` — Bug fixes
- `perf/description` — Performance improvements
- `docs/description` — Documentation-only changes
- `refactor/description` — Code restructuring without behavior changes

### 2. Make Changes

Implement your changes following the standards in this document. Key checkpoints:

- Docstrings added/updated for all new/modified public APIs
- Tests added following the 3-layer test structure
- Type hints present on all public functions and methods
- `make fix` passes without errors
- `make test-all` passes

### 3. Verify Quality

```bash
# Run the full quality gate (do this before every commit)
make fix        # Format + typecheck
make build-all  # Ensure Cython compiles
make test-all   # C + Python tests
```

### 4. Commit and Push

```bash
git add <files>
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

---

## Architecture & Design Patterns

### Protocol-Driven Design

MM Toolbox uses Python `typing.Protocol` to align similar tools behind common interfaces. This allows callers to swap implementations without changing consumer code.

**When to define a protocol:**

- Multiple implementations share the same conceptual interface (e.g., ringbuffer variants, moving average types)
- You want to enforce a contract without inheritance
- Test code should work against any implementation

**Example — `MovingAverageProtocol`:**

```python
# src/mm_toolbox/moving_average/protocol.py
@runtime_checkable
class MovingAverageProtocol(Protocol):
    """Protocol unifying all moving average implementations."""

    def initialize(self, values: npt.NDArray[np.float64]) -> float: ...
    def next(self, new_val: float) -> float: ...
    def update(self, new_val: float) -> float: ...
    def get_value(self) -> float: ...
```

**Guidelines for protocols:**

1. Place protocol definitions in `src/mm_toolbox/<module>/protocol.py`
2. Use `@runtime_checkable` when isinstance checks are needed
3. Document the protocol's purpose and the implementations that satisfy it
4. Keep protocols focused — one concept per protocol file
5. Use `Protocol[T]` generics when the interface is type-parameterized

### Module Structure

New modules follow this structure:

```
src/mm_toolbox/<module_name>/
├── __init__.py          # Public API exports only
├── protocol.py          # Protocol definitions (if multiple implementations)
├── base.pyx             # Base Cython class (if performance-critical)
├── impl_a.pyx           # Implementation A (e.g., SMA, NumericRingBuffer)
├── impl_b.pyx           # Implementation B (e.g., EMA, BytesRingBuffer)
├── helpers.pyx          # Internal helper functions (not public API)
└── types.pxd            # Shared Cython type declarations (if needed)

tests/<module_name>/
├── test_<module>_base.py       # Layer 1 — Primitives
├── test_<module>_impl_a.py     # Layer 2 — Composite/Integration
├── test_<module>_impl_b.py     # Layer 2 — Composite/Integration
└── test_<module>_integration.py # Layer 3 — Mini-Integration
```

**File purposes:**

- `__init__.py`: Only re-export public symbols. Never implement logic here.
- `protocol.py`: Define contracts. No implementation code.
- `*.pyx`: Cython implementations for performance-critical paths.
- `*.pxd`: Cython header files for cross-module type sharing.
- `*.pyi`: Manual type stubs when Cython-generated stubs are insufficient.

### Performance-First Philosophy

This is a high-frequency trading library. Every design decision considers speed and memory:

1. **Cython first**: Computational code lives in `.pyx` files with `boundscheck=False`, `wraparound=False`
2. **Memory efficiency**: Use ringbuffers, avoid intermediate allocations, prefer stack-allocated C structs
3. **Type everything**: Static types help both Cython code generation and maintainability
4. **Profile before optimizing**: Use `cProfile` and line profiler to identify actual bottlenecks

---

## Code Standards

### Import Conventions

Always import `__future__.annotations` in Python files that have docstrings (required by our docstring tooling):

```python
from __future__ import annotations

import os
import sys
from typing import Protocol

import numpy as np

from mm_toolbox.time import time_monotonic_ns
```

**Import order:**

1. `__future__` imports
2. Standard library imports
3. Third-party imports (numpy, etc.)
4. Local/library imports

### Type Hints

- Use strict typing — `ty` (our type checker) runs in strict mode
- Prefer `|` over `typing.Union` (requires `from __future__ import annotations`)
- Use `npt.NDArray[np.float64]` for NumPy arrays
- Annotate all public function signatures, class attributes, and properties
- Cython `.pyx` files: type all `cdef` and `cpdef` signatures explicitly

### Formatting

```bash
make format   # Run ruff format + ruff check --fix
```

- Ruff handles all formatting automatically
- Do not manually format — let the tool do the work
- If ruff complains about something it can't fix, fix it manually

---

## Documentation Standards

### Overview

All source files must have comprehensive documentation. We use three standards depending on the language:


| Language        | Style                   | Tooling Support             |
| --------------- | ----------------------- | --------------------------- |
| Python (`.py`)  | Google-style docstrings | Sphinx, IDE hover           |
| Cython (`.pyx`) | Google-style docstrings | Sphinx (limited), IDE hover |
| C (`.c`, `.h`)  | Javadoc/Doxygen-style   | Doxygen, Clangd             |


### File Headers

Every source file begins with a concise header docstring/comment describing its purpose, usage, and main components.

**Python/Cython:**

```python
"""Brief description of what this module does.

Provides <main functionality>. Includes <key components>.

Main Components:
    - ComponentA: Does X.
    - ComponentB: Does Y.

Typical usage:
    >>> from mm_toolbox.module import ComponentA
    >>> ComponentA()
"""
```

**C/C header:**

```c
/**
 * @file module_name.c
 * @brief Brief description of what this file does.
 *
 * Provides <main functionality>. Includes <key components>.
 */
```

### Classes

Google-style docstrings for all public classes:

```python
class MyClass:
    """One-line summary of the class.

    Extended description if the class has non-trivial behavior
    or important usage notes.

    Attributes:
        attr_a: Description of attr_a.
        attr_b: Description of attr_b.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2. Defaults to None.
    """
```

### Functions

**Standard verbosity** (most functions):

```python
def process_data(data: list[float], threshold: float = 0.5) -> bool:
    """Process incoming market data with a threshold filter.

    Args:
        data: List of price values to process.
        threshold: Minimum value to include. Defaults to 0.5.

    Returns:
        True if any values exceeded the threshold.

    Raises:
        ValueError: If data is empty.
    """
```

**Detailed verbosity** (complex methods with non-obvious behavior):

```python
def normalize_venue(
    self,
    venue_code: str,
    *,
    strict: bool = True,
    fallback: str | None = None,
) -> str:
    """Map a vendor venue code to its canonical identifier.

    Performs case-insensitive lookup against the internal mapping.
    If strict is False and no match exists, returns the input
    unchanged or the fallback value if provided.

    Args:
        venue_code: Raw venue string from the market data feed.
        strict: If True, raise KeyError on unknown venues. If False,
            return the original code or fallback.
        fallback: Value to return when strict=False and lookup fails.
            Ignored when strict=True.

    Returns:
        Canonical venue identifier string.

    Raises:
        KeyError: If venue_code is not in the mapping and strict=True.

    Examples:
        >>> normalizer = VenueNormalizer()
        >>> normalizer.normalize("XNYS")
        'NYSE'
        >>> normalizer.normalize("UNKNOWN", strict=False)
        'UNKNOWN'
    """
```

### Cython-Specific Rules

- `cdef` structs: Document fields with inline comments or a docstring above
- `cdef` classes: Full Google-style docstrings (Sphinx doesn't auto-extract them)
- `cpdef`/`cdef`/`def` methods: Standard Google-style with Args/Returns/Raises
- `.pxd` files: File header docstrings only (no need to duplicate implementation docs)

### C-Specific Rules

- **Functions**: `/** */` with `@brief`, `@param`, `@return`, `@note`
- **Structs**: Document above the declaration with field descriptions
- **Macros**: Brief comment explaining purpose
- **File headers**: Include `@file`, `@brief`, and main components overview

---

## Testing Standards

### Three-Layer Test Architecture

All tests follow a strict layered structure. Each layer gets its own test class.

**Layer 1 — Primitives/Building Blocks:**
Test standalone components in complete isolation (dataclasses, configs, value objects, simple structs).

```python
class TestMovingAverageWindowValidation:
    """Layer 1 — Window size validation across moving average types."""

    def test_window_zero_raises(self):
        """Given window=0, construction raises ValueError."""
        with pytest.raises(ValueError, match="window must be positive"):
            SimpleMovingAverage(window=0)
```

**Layer 2 — Composite Components:**
Test components that consume primitives. Verify integration with dependencies and business logic.

```python
class TestSMAComputation:
    """Layer 2 — Simple moving average computation correctness."""

    def test_sma_computes_correctly(self):
        """Given three values, SMA equals their arithmetic mean."""
        ma = SimpleMovingAverage(window=3)
        result = ma.initialize(np.array([1.0, 2.0, 3.0]))
        assert result == pytest.approx(2.0, abs=1e-12)
```

**Layer 3 — Mini-Integration:**
Test full component integration with realistic usage patterns.

```python
class TestCandlesIntegration:
    """Layer 3 — Full candle aggregation pipeline."""

    def test_process_trade_updates_ohlc(self):
        """Given multiple trades, OHLC reflects correct ranges."""
        candles = TimeBasedCandles(...)
        # ... realistic multi-trade scenario
```

### Test Documentation

Test files use a **test-adapted** Google style. The focus is on *scenario + expected outcome*, not API contracts.

**File headers** state what's under test and the testing strategy:

```python
"""Layer 1 — Primitives tests for moving-average base class behaviour.

Covers window-size validation (edge values: 0, 1, negative, minimum valid),
fast-mode restrictions (history access forbidden), pre-warm error handling
for SMA/WMA, and historical value storage/retrieval.
"""
```

**Test methods** describe Given/When/Then:

```python
def test_process_trade_updates_high_price(self):
    """Processing a trade updates high when price exceeds current.

    Given: Candle with high=100.0
    When:  Trade at price=105.0 is processed
    Then:  Candle.high equals 105.0
    """
```

**Rules:**

- No `Args`/`Returns` sections on test methods
- Explain WHY the edge case matters (what bug does it prevent?)
- Fixtures and helper functions keep standard Google-style with `Args`/`Returns`
- C test comments use `/** @test */` to describe scenarios

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures (keep minimal)
├── <module>/
│   ├── test_<module>_base.py       # Layer 1 tests
│   ├── test_<module>_impl.py       # Layer 2 tests
│   └── test_<module>_integration.py # Layer 3 tests
```

**Guidelines:**

- Each layer gets its own test class
- Test exhaustively: edge cases, boundary values, invalid inputs, error handling
- Do NOT add pointless tests for scenarios that would never occur in practice
- Assume a type checker is always running; skip redundant type validation tests
- Focus on behavior that matters: validation logic, business rules, state transitions

### Running Tests

```bash
make test-all                     # C + Python tests (full suite)
make test-py                      # Python-only tests
make test-c                       # C-only tests
make test-coverage                # Full suite with Cython-aware coverage

# Run specific test modules
uv run pytest tests/candles/ -xvv
uv run pytest tests/moving_average/ -xvv

# Run specific test with output
uv run pytest -k "test_name" -s

# Run specific layer
uv run pytest tests/moving_average/test_moving_average_base.py -xvv
```

---

## Build System

### Essential Make Commands


| Command                 | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| `make fix`              | Run formatter + typechecker (do this before every commit) |
| `make format`           | Run ruff format + ruff check --fix                        |
| `make typecheck`        | Run `ty` type checker on `src/`                           |
| `make test-all`         | Run full C + Python test suite                            |
| `make test-py`          | Run Python tests only                                     |
| `make test-c`           | Run C unit tests only                                     |
| `make build-lib`        | Build Cython extensions in-place                          |
| `make build-test`       | Build Cython test extensions                              |
| `make build-all`        | Build all extensions (lib + test)                         |
| `make rebuild-all`      | Clean + rebuild everything                                |
| `make remove-build-lib` | Remove compiled `.so` files and build artifacts           |
| `make clean-caches`     | Remove `__pycache__`, `.pytest_cache`, `.ruff_cache`      |


### Build Workflow

```bash
# After pulling new code or switching branches
make rebuild-all    # Clean + rebuild lib + test extensions
make test-all       # Verify everything works

# During active development
make build-lib      # Fast incremental build (only changed files)
make test-py        # Quick Python test feedback

# Before committing
make fix            # Format + typecheck
make test-all       # Full verification
```

### Cython Development Tips

For profiling/debugging, enable line tracing:

```python
# In setup.py or tests/setup.py:
compiler_directives = {
    "profile": True,
    "linetrace": True,
}
```

Then rebuild with `CYTHON_TRACE=1`:

```bash
make remove-build-lib
CYTHON_TRACE=1 make build-lib
make test-coverage
```

---

## Commit Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `docs`: Documentation changes
- `style`: Formatting changes (no logic change)
- `chore`: Build/tooling changes

**Scopes:**

- Module name (e.g., `candles`, `orderbook`, `ringbuffer`, `websocket`)
- `logging/standard` or `logging/advanced`
- `orderbook/standard` or `orderbook/advanced`

**Examples:**

```bash
feat(candles): add volume-based candle aggregation
fix(ringbuffer): resolve race condition in MPSC consume
perf(orderbook): optimize ladder updates with SIMD
refactor(logging): unify standard and advanced config parsing
test(moving_average): add boundary tests for window=2
docs: update CONTRIBUTING with protocol guidelines
style: apply ruff formatting to orderbook module
chore(build): update cibuildwheel to 2.19
```

**Rules:**

- Use imperative mood: "add" not "added" or "adds"
- Keep the first line under 72 characters
- Reference issues in the footer: `Closes #123`
- One logical change per commit

---

## Pull Request Process

### Before Opening

1. **Branch is up to date** with the target branch
2. **All quality gates pass:**
  ```bash
   make fix        # Format + typecheck
   make test-all   # Full test suite
  ```
3. **Docstrings added** for all new/modified public APIs
4. **Tests added** following the 3-layer structure
5. **No debug code** left behind (print statements, pdb, etc.)

### PR Description Template

```markdown
## Summary
One-line description of what this PR does.

## Changes
- List specific changes made
- Reference any new protocols or patterns introduced

## Testing
- How was this tested?
- Which layers are covered?
- Any manual verification steps?

## Checklist
- [ ] `make fix` passes
- [ ] `make test-all` passes
- [ ] Docstrings added/updated
- [ ] Tests follow 3-layer structure
- [ ] Protocols updated (if applicable)
```

### Review Criteria

Maintainers review for:

1. **Correctness**: Does the code do what it claims?
2. **Performance**: Any regressions in benchmarks?
3. **Style**: Follows docstring standards, type hints present
4. **Testing**: Layer 1/2/3 coverage, edge cases handled
5. **Architecture**: Uses protocols where appropriate, follows module structure
6. **Documentation**: File headers, class/method docstrings complete

---

## Release Process

### Prerequisites

```bash
# Install all dependencies including build tools
uv sync --all-groups
```

### Building Wheels

```bash
# Development build
make build-lib

# Binary wheel for current platform
make wheel

# Source distribution
make sdist

# PEP 517 clean build
make wheel-pep517

# Validate wheel contains native extensions
make wheel-check
```

### Cross-Platform Wheels

For CI/CD or local cross-platform builds:

```bash
# Install cibuildwheel
uv add --group build cibuildwheel

# Build for current platform
uv run cibuildwheel --platform auto
```

Configured for:

- Python 3.12 and 3.13
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)

### Publishing

**Test PyPI (always test first):**

```bash
make upload-test
```

**Production PyPI (after TestPyPI verification):**

```bash
make upload-prod
```

### Release Checklist

1. [ ] Update version in `pyproject.toml`
2. [ ] Update CHANGELOG
3. [ ] Run `make test-all`
4. [ ] Run `make fix`
5. [ ] Clean build: `make clean-dist && make wheel`
6. [ ] Check distribution: `make check-dist`
7. [ ] Test on TestPyPI: `make upload-test`
8. [ ] Verify installation from TestPyPI
9. [ ] Upload to PyPI: `make upload-prod`
10. [ ] Tag the release: `git tag v1.0.0`
11. [ ] Push tag: `git push origin v1.0.0`
12. [ ] Create GitHub release

---

## Troubleshooting

### Build Issues

**Cython compilation fails:**

```bash
# Ensure dependencies are installed
uv sync --all-groups

# Check Cython version
uv run cython --version  # Should be >= 3.0.11

# Clean and rebuild
make remove-build-all
make build-all
```

**Missing numpy during build:**

```bash
uv sync --all-groups
# numpy must be installed before building extensions
```

**Tests fail with import errors:**

```bash
# Ensure extensions are built
make build-all

# Check PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH
```

### Test Issues

**Specific test hangs (especially logging integration):**

- Some tests use multiprocessing and may hang in resource-constrained environments
- Skip with: `uv run pytest -k "not integration"`

**C tests fail to compile:**

```bash
# Ensure C test build artifacts are clean
make remove-build-tests
make build-test
```

### Wheel Issues

**Wheel missing native extensions:**

```bash
make wheel-check
# If fails, rebuild: make remove-wheel && make wheel
```

**Upload fails:**

- Check PyPI credentials (token-based auth)
- Ensure version number is updated in `pyproject.toml`
- Verify all required metadata is present in `pyproject.toml`

---

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Use GitHub Issues with minimal reproduction
- **Feature requests**: Open an Issue first to discuss approach

By contributing, you agree that your code will be licensed under the MIT License.
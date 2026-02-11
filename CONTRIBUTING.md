# Contributing to MM Toolbox

Welcome! MM Toolbox focuses on high-performance market making tools. Whether you're fixing bugs, adding features, or improving performance, this guide will get you started.

## 🚀 Quick Start

**Prerequisites**: Python 3.12+ and [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/mm-toolbox.git
cd mm-toolbox

# 2. Setup development environment
uv sync --all-groups  # Installs all deps including dev tools
make build-lib        # Compile Cython extensions

# 3. Verify everything works
make test-all       # Run C + Python test suites
make fix            # Format + typecheck
```

## 💡 Making Changes

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes, then verify quality
make fix        # Runs: format + typecheck
make test-all   # Run C + Python tests
make build-lib  # Ensure Cython compiles

# Commit and push
git commit -m "feat: add awesome feature"
git push origin feature/your-feature-name
```

### Code Standards

**Performance First**: This is a high-frequency trading library. Optimize for speed and memory efficiency.

**Code Style**:
- **Ruff formatting**: `make format` (automatic)
- **Type hints required**: Use strict typing for ty compliance
- **Docstrings**: Compact Google style, one-liner for simple functions
- **Dependencies**: Avoid adding new ones unless absolutely necessary

**Testing**:
- **Comprehensive coverage**: Add tests for all new functionality
- **Performance tests**: Include benchmarks for performance-critical code
- **Edge cases**: Test boundary conditions and error handling

### File Structure
```
src/mm_toolbox/
├── module_name/         # New modules here
│   ├── __init__.py     # Public API exports
│   ├── fast.pyx        # Cython implementation (performance-critical)
│   ├── fast.pxd        # Cython headers
│   ├── fast.pyi        # Type stubs (manual, not auto-generated)
│   └── wrapper.py      # Python wrapper (if needed)
tests/module_name/       # Comprehensive test coverage
```

## 🛠️ Development Tips

### Cython Development
```bash
# Fast iteration during development
make build-lib  # Rebuilds only changed files

# For profiling/debugging, enable line tracing:
# Edit setup.py: compiler_directives = {"profile": True, "linetrace": True}
```

### Running Tests
```bash
make test-all                     # C + Python tests
make test-py                      # Python-only tests
make test-c                       # C-only tests
uv run pytest tests/candles/ -xvv  # Specific module
uv run pytest -k "test_name" -s    # Specific test with output
```

### Performance Guidelines
- **Cython first** for computational code (see existing `.pyx` files)
- **Memory efficiency**: Use ringbuffers, avoid unnecessary allocations
- **Type everything**: Helps both performance and maintainability
- **Profile before optimizing**: Use `cProfile` and line profiler

## Building and Releasing

This section covers instructions for building wheels and releasing the package to PyPI.

### Prerequisites

Make sure you have the development and build dependencies installed:

```bash
# Install all dependencies including build tools
uv sync --all-groups

# Or install just the build dependencies
uv sync --group build
```

### Building Extensions

#### Development Build (In-place)

For development, build Cython extensions in-place:

```bash
# Build all Cython extensions
make build-lib

# Or build manually
uv run python setup.py build_ext --inplace --parallel $(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')
```

### Building Wheels

#### Local Wheel Build

To build a wheel for your current platform:

```bash
# Build binary wheel (compiled extensions)
make wheel

# Build source distribution
make sdist

# Build wheel via PEP 517 (clean build isolation)
make wheel-pep517
```

#### Cross-platform Wheels with cibuildwheel

For building wheels across multiple platforms (useful for CI/CD):

```bash
# Install cibuildwheel
uv add --group build cibuildwheel

# Build wheels for all supported platforms
uv run cibuildwheel --platform auto
```

The `pyproject.toml` is configured to build wheels for:
- Python 3.12 and 3.13
- Linux (x86_64 and aarch64)
- macOS (x86_64 and arm64)

### Testing the Build

Before releasing, test the built package:

```bash
# Check the distribution files
make check-dist

# Test installation in a clean environment
uv venv test-env
source test-env/bin/activate
pip install dist/*.whl
python -c "import mm_toolbox; print('Import successful')"
deactivate
rm -rf test-env
```

### Publishing to PyPI

#### Test PyPI (Recommended First)

Always test on TestPyPI first:

```bash
# Upload to TestPyPI
make upload-test

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mm-toolbox
```

#### Production PyPI

Only after testing on TestPyPI:

```bash
# Upload to production PyPI (with safety prompt)
make upload-prod
```

### Release Checklist

Before releasing a new version:

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG** (if you have one)
3. **Run tests**: `make test-all`
4. **Clean build**: `make clean-dist && make wheel`
5. **Check distribution**: `make check-dist`
6. **Test on TestPyPI**: `make upload-test`
7. **Test installation** from TestPyPI
8. **Upload to PyPI**: `make upload-prod`
9. **Tag the release** in git
10. **Create GitHub release** (optional)

### File Inclusion

The following files are included in distributions via `MANIFEST.in`:

- Source code: `*.py`, `*.pyx`, `*.pxd`, `*.pyi`
- C sources: `*.c`, `*.h` (for Cython compilation)
- Documentation: `README.md`, `LICENSE`, `CONTRIBUTING.md`
- Configuration: `pyproject.toml`, `uv.lock`
- Examples and tests: `examples/*.py`, `tests/*.py`

### Troubleshooting

#### Common Issues

1. **Build fails on missing dependencies**:
   ```bash
   uv sync --all-groups
   ```

2. **Cython compilation errors**:
   - Check that Cython version is >=3.0.11
   - Ensure numpy is installed
   - Try cleaning: `make remove-build-lib && make build-lib`

3. **Wheel contains wrong files**:
   - Check `MANIFEST.in`
   - Clean and rebuild: `make remove-wheel && make wheel`

4. **Upload fails**:
   - Check PyPI credentials
   - Ensure version number is updated
   - Verify all required metadata is present

#### Beta Release Playbook (Used for 1.0.0b4)

If you are shipping a beta and need Linux + macOS wheels quickly, this is the exact flow that worked:

1. **Run a full local rebuild before tests**:
   ```bash
   make rebuild-all
   make test-all
   make fix
   ```

2. **Trigger release workflow on a non-default branch via CLI** (useful when Actions UI snaps back to `main`):
   ```bash
   gh workflow run release.yml --ref v1.0b2 -f environment=testpypi
   gh run list --workflow release.yml --branch v1.0b2 --limit 10
   gh run watch <run-id>
   gh run view <run-id> --log-failed
   ```

3. **Known Linux CI hang pattern**:
   - We observed Ubuntu wheel jobs hang/timeout in:
     - `tests/logging/advanced/test_advanced_integration.py`
     - commonly around `TestIntegration::test_multiple_workers` while waiting on `p.join()`
   - This blocked Linux wheel publication from CI in some runs.

4. **Local Linux wheel fallback with cibuildwheel** (manylinux x86_64):
   ```bash
   # Optional: remove stale local IPC sockets to reduce archive warnings
   find .ipc -type s -delete 2>/dev/null || true

   CIBW_TEST_COMMAND='python -c "import mm_toolbox"' \
   uv tool run cibuildwheel --platform linux --archs x86_64 --output-dir dist
   ```
   - The `CIBW_TEST_COMMAND` override keeps a smoke test (`import mm_toolbox`) and avoids full test-suite hangs during wheel build.
   - Keeping all artifacts in `dist/` avoids a second wheel staging directory.

5. **Upload wheels directly (skip existing files)**:
   ```bash
   set -a && source .env && set +a

   # TestPyPI
   uv run python -m twine upload \
     --repository-url https://test.pypi.org/legacy/ \
     -u __token__ -p "$TEST_PUBLISH_TOKEN" \
     --skip-existing dist/*

   # PyPI
   uv run python -m twine upload \
     --repository-url https://upload.pypi.org/legacy/ \
     -u __token__ -p "$PUBLISH_TOKEN" \
     --skip-existing dist/*
   ```

6. **Verify artifacts after upload**:
   ```bash
   uv run python - <<'PY'
   import json, urllib.request
   with urllib.request.urlopen("https://pypi.org/pypi/mm-toolbox/1.0.0b4/json", timeout=30) as r:
       data = json.load(r)
   for f in data["urls"]:
       print(f["filename"])
   PY
   ```

7. **Pre-release install note**:
   ```bash
   pip install --pre mm-toolbox==1.0.0b4
   ```
   - Pre-release versions are excluded by default unless `--pre` is provided.

### Environment Variables

For automated builds, you can set:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your-pypi-api-token
export TWINE_REPOSITORY_URL=https://upload.pypi.org/legacy/  # for production
```

### CI/CD Integration

The project is ready for GitHub Actions or other CI systems. Example workflow for building wheels:

```yaml
# .github/workflows/build.yml
name: Build Wheels

on: [push, pull_request]

jobs:
  build_wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install uv
      run: pip install uv
    
    - name: Build wheels
      run: uv run cibuildwheel --output-dir dist
    
    - uses: actions/upload-artifact@v3
      with:
        path: ./dist/*.whl
```

## 📋 Contribution Types

**🐛 Bug Fixes**: Include reproduction steps and tests that fail before your fix.

**✨ New Features**: 
- Open an issue first to discuss the approach
- Include comprehensive tests and benchmarks
- Update type stubs (`.pyi` files) manually
- Add examples in docstrings

**⚡ Performance Improvements**: 
- Include benchmarks showing improvement
- Maintain backward compatibility
- Consider memory usage, not just speed

**📚 Documentation**: Keep it concise but complete.

## 🔍 Review Process

1. **CI must pass**: All tests, linting, and type checking
2. **Performance regression**: Benchmarks should not degrade
3. **Code review**: Maintainer will review for style and correctness
4. **Testing**: New code needs test coverage

## 📝 Commit Guidelines

Use conventional commits:
```bash
feat: add exponential moving average implementation
fix: resolve memory leak in ringbuffer
perf: optimize candle aggregation by 25%
test: add edge cases for price rounding
docs: update API examples
```

## 🆘 Getting Help

- **Questions**: Open a GitHub discussion
- **Bugs**: Use GitHub issues with minimal reproduction
- **Features**: Discuss in issues before implementing

## 📄 License

By contributing, you agree your code will be licensed under MIT.

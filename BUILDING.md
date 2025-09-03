# Building and Releasing MM Toolbox

This document contains instructions for building wheels and releasing the package to PyPI.

## Prerequisites

Make sure you have the development and build dependencies installed:

```bash
# Install all dependencies including build tools
uv sync --all-groups

# Or install just the build dependencies
uv sync --group build
```

## Building Extensions

### Development Build (In-place)

For development, build Cython extensions in-place:

```bash
# Build all Cython extensions
make build

# Or build manually
uv run python setup.py build_ext --inplace --parallel $(nproc)
```

## Building Wheels

### Local Wheel Build

To build a wheel for your current platform:

```bash
# Build wheel (cleans first)
make wheel

# Build source distribution
make sdist

# Or build both
uv run python -m build
```

### Cross-platform Wheels with cibuildwheel

For building wheels across multiple platforms (useful for CI/CD):

```bash
# Install cibuildwheel
uv add --group build cibuildwheel

# Build wheels for all supported platforms
uv run cibuildwheel --platform auto
```

The `pyproject.toml` is configured to build wheels for:
- Python 3.12 and 3.13
- Linux (x86_64, excluding musllinux)
- macOS (x86_64 and arm64)
- Windows (x86_64)

## Testing the Build

Before releasing, test the built package:

```bash
# Check the distribution files
make check-dist

# Test installation in a clean environment
uv venv test-env
source test-env/bin/activate  # or test-env\Scripts\activate on Windows
pip install dist/*.whl
python -c "import mm_toolbox; print('Import successful')"
deactivate
rm -rf test-env
```

## Publishing to PyPI

### Test PyPI (Recommended First)

Always test on TestPyPI first:

```bash
# Upload to TestPyPI
make upload-test

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mm-toolbox
```

### Production PyPI

Only after testing on TestPyPI:

```bash
# Upload to production PyPI (with safety prompt)
make upload-prod
```

## Release Checklist

Before releasing a new version:

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG** (if you have one)
3. **Run tests**: `make test`
4. **Clean build**: `make wheel-clean && make wheel`
5. **Check distribution**: `make check-dist`
6. **Test on TestPyPI**: `make upload-test`
7. **Test installation** from TestPyPI
8. **Upload to PyPI**: `make upload-prod`
9. **Tag the release** in git
10. **Create GitHub release** (optional)

## File Inclusion

The following files are included in distributions via `MANIFEST.in`:

- Source code: `*.py`, `*.pyx`, `*.pxd`, `*.pyi`
- C sources: `*.c`, `*.h` (for Cython compilation)
- Documentation: `README.md`, `LICENSE`, `CONTRIBUTING.md`
- Configuration: `pyproject.toml`, `uv.lock`
- Examples and tests: `examples/*.py`, `tests/*.py`

## Troubleshooting

### Common Issues

1. **Build fails on missing dependencies**:
   ```bash
   uv sync --all-groups
   ```

2. **Cython compilation errors**:
   - Check that Cython version is >=3.0.11
   - Ensure numpy is installed
   - Try cleaning: `make remove-build && make build`

3. **Wheel contains wrong files**:
   - Check `MANIFEST.in`
   - Clean and rebuild: `make wheel-clean && make wheel`

4. **Upload fails**:
   - Check PyPI credentials
   - Ensure version number is updated
   - Verify all required metadata is present

### Environment Variables

For automated builds, you can set:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=your-pypi-api-token
export TWINE_REPOSITORY_URL=https://upload.pypi.org/legacy/  # for production
```

## CI/CD Integration

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
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install uv
      run: pip install uv
    
    - name: Build wheels
      run: uv run cibuildwheel --output-dir wheelhouse
    
    - uses: actions/upload-artifact@v3
      with:
        path: ./wheelhouse/*.whl
```

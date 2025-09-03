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
make build           # Compile Cython extensions

# 3. Verify everything works
make test           # Run test suite
make check         # Format, lint, typecheck
```

## 💡 Making Changes

### Development Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes, then verify quality
make check    # Runs: format, lint, typecheck
make test     # Run all tests
make build    # Ensure Cython compiles

# Commit and push
git commit -m "feat: add awesome feature"
git push origin feature/your-feature-name
```

### Code Standards

**Performance First**: This is a high-frequency trading library. Optimize for speed and memory efficiency.

**Code Style**:
- **Ruff formatting**: `make format` (automatic)
- **Type hints required**: Use strict typing for Pyright compliance
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
make build  # Rebuilds only changed files

# For profiling/debugging, enable line tracing:
# Edit setup.py: compiler_directives = {"profile": True, "linetrace": True}
```

### Running Tests
```bash
make test                    # All tests
uv run pytest tests/candles/ -xvv  # Specific module
uv run pytest -k "test_name" -s    # Specific test with output
```

### Performance Guidelines
- **Cython first** for computational code (see existing `.pyx` files)
- **Memory efficiency**: Use ringbuffers, avoid unnecessary allocations
- **Type everything**: Helps both performance and maintainability
- **Profile before optimizing**: Use `cProfile` and line profiler

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

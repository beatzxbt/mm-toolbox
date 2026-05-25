.PHONY: help format typecheck fix test-py test-c test-all test-coverage sync build rebuild remove-build clean \
        wheel wheel-pep517 wheel-check sdist dist check-dist clean-dist upload-test \
        macos-wheels linux-wheels-x86_64 linux-wheels-aarch64 linux-wheels wheels %

.DEFAULT_GOAL := help

TEST_FLAGS := -xvv -s -p no:anchorpy

format: ## Format code using ruff
	uv run ruff format .
	uv run ruff check --fix --unsafe-fixes .

typecheck: ## Run static type checking
	uv run ty check src/

fix: ## Run all formatters and typecheck
	$(MAKE) format typecheck

test-py: ## Run tests
	PYTHONPATH=src uv run pytest $(TEST_FLAGS)

test-c: ## Run C unit tests
	$(MAKE) -C tests/orderbook/advanced/c test

test-all: build ## Run all tests (C + Python)
	$(MAKE) test-c test-py

test-coverage: ## Run tests with Cython-aware coverage (rebuild, test, clean)
	$(MAKE) remove-build-lib
	CYTHON_TRACE=1 uv run python setup.py build_ext --inplace --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')
	@PYTHONPATH=src uv run pytest --cov=src --cov-report=term-missing --cov-report=html; \
	pytest_exit=$$?; \
	find ./src -name "*.c" -type f | while read f; do [ -f "$${f%.c}.pyx" ] && rm -f "$$f"; done; \
	exit $$pytest_exit

sync: ## Re‑lock and install latest versions
	uv lock --upgrade       # rebuild uv.lock with newer pins
	uv sync --all-groups    # install everything into .venv

build-lib:
	uv run python setup.py build_ext --inplace --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')

build-test:
	$(MAKE) -C tests/orderbook/advanced/c test
	cd tests && uv run python setup.py build_ext --inplace --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')

build: ## Build all Cython extensions
	$(MAKE) build-lib build-test

clean: ## Remove all caches (pycache, pytest, ruff, uv, cibw)
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ .ruff_cache/ htmlcov/ .cibw-cache/
	uv cache clean

remove-build-lib:
	rm -rf build/ *.egg-info/
	find ./src -name "*.so" -delete
	$(MAKE) clean

remove-build-tests:
	rm -rf build/ *.egg-info/
	find ./tests/orderbook/advanced/c -name "*.so" -delete
	cd tests && uv run python setup.py clean --all || true
	find ./tests/orderbook/advanced -path "*/engine/*.so" -delete
	find ./tests/orderbook/advanced -path "*/engine/*.c" -type f -delete
	find ./tests/orderbook/advanced -path "*/wrapper_cython/*.so" -delete
	find ./tests/orderbook/advanced -path "*/wrapper_cython/*.c" -type f -delete
	find ./tests -name "cython_test_*.so" -delete
	find ./tests -name "cython_test_*.c" -type f -delete
	$(MAKE) clean

remove-build: ## Remove build artifacts and compiled extensions
	$(MAKE) remove-build-lib remove-build-tests

rebuild: remove-build build ## Clean and rebuild all Cython extensions

wheel: ## Build binary wheel distribution
	$(MAKE) remove-build-lib
	$(MAKE) build-lib
	uv run python setup.py bdist_wheel

wheel-pep517: ## Build wheel via PEP 517
	$(MAKE) remove-build-lib
	uv run python -m build --wheel

wheel-check: ## Validate wheel contains native extensions
	uv run python -c "import glob,zipfile,sys,os; wheels=glob.glob('dist/*.whl'); \
    wheels.sort(key=os.path.getmtime); \
    whl=wheels[-1] if wheels else None; \
    (whl and any(n.endswith(('.so','.pyd')) for n in zipfile.ZipFile(whl).namelist())) \
        or sys.exit('wheel missing native extensions'); \
    print(f'wheel ok: {whl}')"

sdist: ## Build source distribution
	$(MAKE) remove-build-lib
	uv run python -m build --sdist

dist: ## Build wheel and sdist
	$(MAKE) wheel sdist

check-dist: ## Check distribution files for PyPI upload
	uv run python -m twine check dist/*

clean-dist: ## Remove distribution artifacts
	if [ -d dist ]; then find dist -type f ! -name ".gitignore" -delete; fi

upload-test: ## Upload to TestPyPI
	uv run python -m twine upload --repository testpypi dist/*

# Platform-specific wheel builds (for parallel execution)
macos-wheels: ## Build macOS wheels
	uv run cibuildwheel --platform macos

linux-wheels-x86_64: ## Build Linux x86_64 wheels
	CIBW_ARCHS_LINUX=x86_64 uv run cibuildwheel --platform linux

linux-wheels-aarch64: ## Build Linux aarch64 wheels (longer timeout for QEMU)
	CIBW_ARCHS_LINUX=aarch64 timeout 1800 uv run cibuildwheel --platform linux

linux-wheels: ## Build all Linux wheels in parallel (run with: make -j2 linux-wheels)
	$(MAKE) -j2 linux-wheels-x86_64 linux-wheels-aarch64

wheels: ## Build all wheels (macOS + Linux)
	$(MAKE) macos-wheels linux-wheels

# Pattern rule so additional args do not trigger "No rule to make target"
%:
	@:

help: ## Display this help message
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Code Quality:'
	@grep -E '^(format|typecheck|fix):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Testing:'
	@grep -E '^(test-py|test-c|test-all|test-coverage):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Build:'
	@grep -E '^(build|rebuild|remove-build):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Distribution:'
	@grep -E '^(wheel|wheel-pep517|wheel-check|sdist|dist|check-dist|clean-dist):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Deployment:'
	@grep -E '^(upload-test):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Other:'
	@grep -E '^(sync|clean|help):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

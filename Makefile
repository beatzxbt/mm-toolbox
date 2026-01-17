.PHONY: help format typecheck fix test-py test-c test-all sync build-lib build-test build-all \
        remove-build-lib remove-build-tests rebuild-test remove-build-all rebuild-all wheel wheel-binary wheel-pep517 \
        wheel-check remove-wheel sdist remove-sdist check-dist clean-dist upload-test upload-prod clean-caches %

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

test-all: ## Run all tests (C + Python)
	$(MAKE) test-c test-py 

sync: ## Re‑lock and install latest versions
	uv lock --upgrade       # rebuild uv.lock with newer pins
	uv sync --all-groups    # install everything into .venv

build-lib: ## Build Cython extensions in-place
	uv run python setup.py build_ext --inplace --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')

build-test: ## Build all test extensions (C unit tests + Cython)
	$(MAKE) -C tests/orderbook/advanced/c test
	cd tests && uv run python setup.py build_ext --inplace --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')

build-all: ## Build all Cython extensions
	$(MAKE) build-lib build-test

clean-caches: ## Remove pytest and ruff caches
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ .ruff_cache/

clean-dist: ## Remove distribution artifacts
	if [ -d dist ]; then find dist -type f ! -name ".gitignore" -delete; fi

remove-build-lib: ## Remove build artifacts and compiled extensions
	rm -rf build/ *.egg-info/
	find ./src -name "*.so" -delete
	$(MAKE) clean-caches

remove-build-tests: ## Remove C test build artifacts
	rm -rf build/ *.egg-info/
	find ./tests/orderbook/advanced/c -name "*.so" -delete
	cd tests && uv run python setup.py clean --all || true
	find ./tests/orderbook/advanced -path "*/cython/*.so" -delete
	find ./tests/orderbook/advanced -path "*/cython/*.c" -type f -delete
	$(MAKE) clean-caches

remove-build-all: ## Remove build artifacts and compiled extensions
	$(MAKE) remove-build-lib remove-build-tests

rebuild-lib: remove-build-lib build-lib ## Clean and rebuild Cython extensions

rebuild-test: remove-build-tests build-test ## Clean and rebuild Cython test extensions

rebuild-all: remove-build-all build-all ## Clean and rebuild all Cython extensions

wheel: ## Build binary wheel distribution
	$(MAKE) wheel-binary

wheel-binary: ## Build binary wheel distribution (compiled extensions)
	$(MAKE) remove-build-lib
	$(MAKE) build-lib
	uv run python setup.py bdist_wheel

wheel-pep517: ## Build wheel via PEP 517 (uv_build)
	$(MAKE) remove-build-lib
	uv run python -m build --wheel

remove-wheel: ## Clean wheel build artifacts
	$(MAKE) remove-build-lib

sdist: ## Build source distribution
	$(MAKE) remove-build-lib
	uv run python -m build --sdist

remove-sdist: ## Clean sdist build artifacts
	$(MAKE) remove-build-lib

check-dist: ## Check distribution files for PyPI upload
	uv run python -m twine check dist/*

wheel-check: ## Validate wheel contains native extensions
	uv run python -c "import glob,zipfile,sys,os; wheels=glob.glob('dist/*.whl'); \
    wheels.sort(key=os.path.getmtime); \
    whl=wheels[-1] if wheels else None; \
    (whl and any(n.endswith(('.so','.pyd')) for n in zipfile.ZipFile(whl).namelist())) \
        or sys.exit('wheel missing native extensions'); \
    print(f'wheel ok: {whl}')"

upload-test: ## Upload to TestPyPI
	uv run python -m twine upload --repository testpypi dist/*

upload-prod: ## Upload to PyPI (PRODUCTION)
	@echo "WARNING: This will upload to PRODUCTION PyPI!"
	@echo "Make sure you have:"
	@echo "  1. Updated version number"
	@echo "  2. Built fresh wheel: make wheel"
	@echo "  3. Tested the package thoroughly"
	@echo ""
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	uv run python -m twine upload dist/*

# Pattern rule so additional args do not trigger "No rule to make target"
%:
	@:

# Project‑specific entry points (replace poetry run → uv run)

help: ## Display this help message
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Code Quality:'
	@grep -E '^(format|typecheck|fix):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Testing:'
	@grep -E '^(test-py|test-c|test-all):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Build:'
	@grep -E '^(build-lib|build-test|build-all|remove-build-lib|remove-build-tests|rebuild-test|remove-build-all|rebuild-all|clean-caches):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Distribution:'
	@grep -E '^(wheel|wheel-binary|wheel-pep517|wheel-check|remove-wheel|sdist|remove-sdist|check-dist|clean-dist):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Deployment:'
	@grep -E '^(upload-test|upload-prod):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Other:'
	@grep -E '^(sync|help):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: help install format lint lint-fix lint-check typecheck docstring style fix \
        check sync clean test stats watch live-stream run-client reclaim \
        monitor monitor-charts build watch-build wheel wheel-clean sdist \
        upload-test upload-prod check-dist
.DEFAULT_GOAL := help

TEST_FLAGS := -xvv -s -p no:anchorpy

format: ## Format code using ruff
	uv run ruff format .

lint: ## Run linting using ruff
	uv run ruff check --fix .
	uv run ruff check --fix --unsafe-fixes .

typecheck: ## Run static type checking
	uv run pyright

docstring: ## Check docstring style and completeness
	uv run ruff check . --select D,PD

docstring-fix: ## Fix docstring style and completeness
	uv run ruff check --fix . --select D,PD
	uv run ruff check --fix --unsafe-fixes . --select D,PD

style: ## Check code style (without docstrings)
	uv run ruff check . --select E,W,F,I,N,UP,B,C4,SIM,TCH

style-fix: ## Fix code style (without docstrings)
	uv run ruff check --fix . --select E,W,F,I,N,UP,B,C4,SIM,TCH
	uv run ruff check --fix --unsafe-fixes . --select E,W,F,I,N,UP,B,C4,SIM,TCH

fix: ## Run all formatters and fixers
	$(MAKE) format lint-fix style-fix 

check: ## Run all formatters, fixers, and typecheck
	$(MAKE) fix typecheck 

sync: ## Re‑lock and install latest versions
	uv lock --upgrade       # rebuild uv.lock with newer pins
	uv sync --all-groups    # install everything into .venv

test: ## Run tests
	PYTHONPATH=src uv run pytest $(TEST_FLAGS)

build: ## Build Cython extensions in-place
	uv run python setup.py build_ext --inplace --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')

remove-build: ## Remove build artifacts and compiled extensions
	rm -rf build/ *.egg-info/
	find ./src -name "*.so" -delete

stats: ## Show code quality statistics
	@echo "=== Docstring Coverage ==="
	@uv run ruff check . --select D --statistics
	@echo "\n=== Missing Type Hints ==="
	@uv run  ruff check . --select ANN --statistics
	@echo "\n=== Style Issues ==="
	@uv run  ruff check . --select E,W,F,I,N --statistics


# Watch tests, optionally limited to a path:
#   make watch                → watch entire suite
#   make watch path/to/file   → watch single test file
watch:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		PYTHONPATH=src uv run pytest-watcher . --runner "pytest tests $(TEST_FLAGS) -W ignore::DeprecationWarning"; \
	else \
		path_arg="$(filter-out $@,$(MAKECMDGOALS))"; \
		PYTHONPATH=src uv run pytest-watcher . --runner "pytest src/tests/$$path_arg $(TEST_FLAGS) -W ignore::DeprecationWarning"; \
	fi

watch-build: ## Rebuild Cython extensions on source changes
	uv run python scripts/watch_build.py --parallel $$(uv run python -c 'import os;print(max(1,(os.cpu_count() or 2)-1))')

wheel: ## Build wheel distribution
	$(MAKE) remove-build
	uv run python -m build --wheel

sdist: ## Build source distribution
	$(MAKE) remove-build
	uv run python -m build --sdist

wheel-clean: ## Clean wheel build artifacts
	rm -rf build/ dist/ *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + || true

check-dist: ## Check distribution files for PyPI upload
	uv run python -m twine check dist/*

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
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

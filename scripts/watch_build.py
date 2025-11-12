"""Watch Cython sources and rebuild extensions on changes.

Usage:
  uv run python scripts/watch_build.py --parallel 8

This watches `.pyx`, `.pxd`, `.pyi`, `.c`, `.h` files under `src/mm_toolbox/`
and runs `setup.py build_ext --inplace` with the requested parallelism.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from watchfiles import Change, DefaultFilter, watch

CYTHON_EXTENSIONS: tuple[str, ...] = (".pyx", ".pxd", ".pyi", ".c", ".h")


class CythonFilter(DefaultFilter):
    """Filter only relevant Cython-related files and ignore build dirs."""

    def __call__(self, change: Change, path: str) -> bool:  # noqa: D401
        if not super().__call__(change, path):
            return False
        p = Path(path)
        if any(
            part
            in {".git", ".ruff_cache", ".pytest_cache", "build", "dist", "__pycache__"}
            for part in p.parts
        ):
            return False
        return p.suffix in CYTHON_EXTENSIONS


def run_build(parallel: int) -> int:
    """Run a single build. Returns process return code.

    Args:
        parallel: Number of parallel compile jobs.

    """
    cmd = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
        "--parallel",
        str(parallel),
    ]
    print(f"[watch-build] Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(proc.stdout, end="")
    if proc.returncode == 0:
        print("[watch-build] Build succeeded.")
    else:
        print(f"[watch-build] Build failed with code {proc.returncode}.")
    return proc.returncode


def watch_and_build(
    paths: Iterable[str], parallel: int, debounce_ms: int, initial: bool
) -> None:
    """Watch paths and trigger builds on changes.

    Args:
        paths: Directories to watch.
        parallel: Number of parallel compile jobs.
        debounce_ms: Debounce window in milliseconds.
        initial: Whether to run an initial build immediately.

    """
    if initial:
        run_build(parallel)

    debounce_s = max(0, debounce_ms) / 1000.0
    last_built_at = 0.0

    print(f"[watch-build] Watching: {', '.join(paths)}")
    for changes in watch(*paths, recursive=True, watch_filter=CythonFilter()):
        now = time.monotonic()
        if now - last_built_at < debounce_s:
            continue
        changed_files = ", ".join(sorted({p for _, p in changes}))
        print(f"[watch-build] Detected changes in: {changed_files}")
        run_build(parallel)
        last_built_at = time.monotonic()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    default_parallel = max(1, (os.cpu_count() or 2) - 1)
    parser = argparse.ArgumentParser(
        description="Watch Cython sources and rebuild on changes."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["src/mm_toolbox"],
        help="One or more directories to watch (default: src/mm_toolbox)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=default_parallel,
        help=f"Number of parallel compile jobs (default: {default_parallel})",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=250,
        help="Debounce window in milliseconds (default: 250)",
    )
    parser.add_argument(
        "--no-initial",
        action="store_true",
        help="Do not run an initial build on startup",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Entry point."""
    args = parse_args()
    try:
        watch_and_build(
            paths=args.paths,
            parallel=args.parallel,
            debounce_ms=args.debounce,
            initial=not args.no_initial,
        )
        return 0
    except KeyboardInterrupt:
        print("[watch-build] Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

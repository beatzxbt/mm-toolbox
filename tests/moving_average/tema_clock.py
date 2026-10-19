"""Thin Python wrapper for deterministic native TEMA clock helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cython_test_tema_clock import (
    last_time_ns,
    next_at,
    set_time_ns,
    update_at,
)

__all__ = ["last_time_ns", "next_at", "set_time_ns", "update_at"]

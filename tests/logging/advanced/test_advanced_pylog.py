"""Layer 1 — Primitives tests for Python-accessible log types.

Covers ``PyLogLevel`` enum semantics (numeric values, ordering helpers) and
``PyLog`` struct behaviour (creation, iteration, indexing, equality).
"""

from __future__ import annotations

import pytest

from mm_toolbox.logging.advanced.pylog import PyLog, PyLogLevel


class TestPyLogLevel:
    """Layer 1 — ``PyLogLevel`` primitive tests."""

    def test_level_values(self):
        """Given the enum members, their numeric values increase with severity."""
        assert PyLogLevel.TRACE == 0
        assert PyLogLevel.DEBUG == 1
        assert PyLogLevel.INFO == 2
        assert PyLogLevel.WARNING == 3
        assert PyLogLevel.ERROR == 4

    def test_is_lower(self):
        """Given two levels, ``is_lower`` returns True only when the caller is less severe."""
        assert PyLogLevel.DEBUG.is_lower(PyLogLevel.INFO) is True
        assert PyLogLevel.INFO.is_lower(PyLogLevel.DEBUG) is False
        assert PyLogLevel.TRACE.is_lower(PyLogLevel.ERROR) is True

    def test_is_higher(self):
        """Given two levels, ``is_higher`` returns True only when the caller is more severe."""
        assert PyLogLevel.WARNING.is_higher(PyLogLevel.INFO) is True
        assert PyLogLevel.INFO.is_higher(PyLogLevel.WARNING) is False
        assert PyLogLevel.ERROR.is_higher(PyLogLevel.TRACE) is True


class TestPyLog:
    """Layer 1 — ``PyLog`` struct primitive tests."""

    def test_iteration(self):
        """Given a ``PyLog``, iterating it yields the four fields in order."""
        log = PyLog(1, b"name", PyLogLevel.DEBUG, b"msg")
        items = list(log)
        assert items == [1, b"name", PyLogLevel.DEBUG, b"msg"]

    def test_getitem_valid_indices(self):
        """Given indices 0–3, each returns the corresponding field."""
        log = PyLog(1, b"name", PyLogLevel.DEBUG, b"msg")
        assert log[0] == 1
        assert log[1] == b"name"
        assert log[2] == PyLogLevel.DEBUG
        assert log[3] == b"msg"

    def test_getitem_index_error(self):
        """Given an out-of-range index, ``__getitem__`` raises ``IndexError``."""
        log = PyLog(1, b"name", PyLogLevel.DEBUG, b"msg")
        with pytest.raises(IndexError, match="out of range"):
            _ = log[4]
        with pytest.raises(IndexError, match="out of range"):
            _ = log[-1]

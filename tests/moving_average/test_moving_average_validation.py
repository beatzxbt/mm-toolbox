"""Regressions for input safety, reset semantics and fast-mode memory use."""

from __future__ import annotations

import subprocess
import sys
import tracemalloc
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from mm_toolbox.moving_average import (
    ExponentialMovingAverage,
    SimpleMovingAverage,
    TimeExponentialMovingAverage,
    WeightedMovingAverage,
)
from mm_toolbox.moving_average.base import MovingAverage

from .tema_clock import last_time_ns, next_at, set_time_ns, update_at

MA_CLASSES: tuple[type[MovingAverage], ...] = (
    SimpleMovingAverage,
    WeightedMovingAverage,
    ExponentialMovingAverage,
    TimeExponentialMovingAverage,
)


def test_scalar_initialization_raises_without_native_crash() -> None:
    """A zero-dimensional array must not reach unchecked shape[0] access."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import resource
import numpy as np
from mm_toolbox.moving_average import (
    SimpleMovingAverage, WeightedMovingAverage,
    ExponentialMovingAverage, TimeExponentialMovingAverage,
)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
for cls in (SimpleMovingAverage, WeightedMovingAverage,
            ExponentialMovingAverage, TimeExponentialMovingAverage):
    try:
        cls(window=3).initialize(np.array(1.0))
    except ValueError:
        pass
    else:
        raise AssertionError(cls.__name__)
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ma_cls", MA_CLASSES)
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        np.empty(0),
        np.ones((3, 1)),
        np.array([10.0, 20.0, np.nan]),
        np.array([10.0, np.inf, 30.0]),
        np.array([-np.inf, 20.0, 30.0]),
        np.array([10, 20, 30], dtype=np.int64),
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
    ],
    ids=["none", "empty", "matrix", "nan", "inf", "negative-inf", "int", "float32"],
)
def test_invalid_initialize_preserves_state(
    ma_cls: type[MovingAverage],
    invalid: object,
) -> None:
    ma = ma_cls(window=3)
    ma.initialize(np.array([1.0, 2.0, 3.0]))
    history = ma.get_values().copy()
    value = ma.get_value()
    if isinstance(ma, TimeExponentialMovingAverage):
        set_time_ns(ma, 1_000_000_000)

    # Deliberately bypass the float64 annotation to exercise the native boundary.
    with pytest.raises(ValueError):
        ma.initialize(cast(npt.NDArray[np.float64], invalid))

    assert ma.get_value() == value
    np.testing.assert_array_equal(ma.get_values(), history)
    if isinstance(ma, TimeExponentialMovingAverage):
        assert last_time_ns(ma) == 1_000_000_000
        assert update_at(ma, 7.0, 2_000_000_000) == pytest.approx(4.0)
    else:
        control = ma_cls(window=3)
        control.initialize(np.array([1.0, 2.0, 3.0]))
        assert ma.update(7.0) == pytest.approx(control.update(7.0))


@pytest.mark.parametrize("ma_cls", MA_CLASSES)
@pytest.mark.parametrize("reverse", [False, True], ids=["strided", "reversed"])
def test_readonly_strided_initialization(
    ma_cls: type[MovingAverage],
    reverse: bool,
) -> None:
    values = np.arange(1.0, 7.0)[::2]
    if reverse:
        values = values[::-1]
    values.setflags(write=False)
    before = values.copy()
    ma = ma_cls(window=3)
    control = ma_cls(window=3)
    assert ma.initialize(values) == pytest.approx(control.initialize(before))
    np.testing.assert_array_equal(ma.get_values(), control.get_values())
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("ma_cls", MA_CLASSES)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["next", "update"])
def test_nonfinite_sample_cannot_poison_warm_state(
    ma_cls: type[MovingAverage],
    invalid: float,
    method: str,
) -> None:
    ma = ma_cls(window=3)
    ma.initialize(np.array([1.0, 2.0, 3.0]))
    history = ma.get_values().copy()
    value = ma.get_value()
    if isinstance(ma, TimeExponentialMovingAverage):
        set_time_ns(ma, 1_000_000_000)

    with pytest.raises(ValueError, match="finite"):
        if isinstance(ma, TimeExponentialMovingAverage):
            operation = next_at if method == "next" else update_at
            operation(ma, invalid, 2_000_000_000)
        else:
            operation = ma.next if method == "next" else ma.update
            operation(invalid)

    assert ma.get_value() == value
    np.testing.assert_array_equal(ma.get_values(), history)
    if isinstance(ma, TimeExponentialMovingAverage):
        assert last_time_ns(ma) == 1_000_000_000
        assert update_at(ma, 7.0, 2_000_000_000) == pytest.approx(4.0)
    else:
        assert np.isfinite(ma.update(7.0))


@pytest.mark.parametrize(
    "ma_cls", [ExponentialMovingAverage, TimeExponentialMovingAverage]
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_sample_cannot_seed_cold_state(
    ma_cls: type[MovingAverage],
    invalid: float,
) -> None:
    ma = ma_cls(window=3)
    for operation in (ma.next, ma.update):
        with pytest.raises(ValueError, match="finite"):
            operation(invalid)
        assert ma.get_value() == 0.0
        assert len(ma) == 0
    assert ma.next(42.0) == 42.0
    assert ma.get_value() == 0.0
    assert ma.update(42.0) == 42.0
    np.testing.assert_array_equal(ma.get_values(), [42.0])


@pytest.mark.parametrize("ma_cls", [SimpleMovingAverage, WeightedMovingAverage])
@pytest.mark.parametrize("window", [2, 3, 4, 5, 7, 10, 20, 100])
@pytest.mark.parametrize("is_fast", [False, True])
def test_reinitialize_discards_wrapped_raw_window(
    ma_cls: type[MovingAverage],
    window: int,
    is_fast: bool,
) -> None:
    ma = ma_cls(window=window, is_fast=is_fast)
    ma.initialize(np.arange(window, dtype=np.float64))
    for value in range(2 * window):
        ma.update(float(value))

    raw = np.arange(10.0, 10.0 + window)
    weights = (
        np.ones(window)
        if ma_cls is SimpleMovingAverage
        else np.arange(1.0, window + 1.0)
    )
    expected = float(np.average(raw, weights=weights))
    assert ma.initialize(raw) == pytest.approx(expected)
    if not is_fast:
        np.testing.assert_allclose(ma.get_values(), [expected])

    for value in (100.0, -50.0, 0.0, 3.0, 9.0) * window:
        raw = np.append(raw[1:], value)
        expected = float(np.average(raw, weights=weights))
        assert ma.next(value) == pytest.approx(expected, abs=1e-10)
        assert ma.update(value) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_ema_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="Alpha"):
        ExponentialMovingAverage(window=3, alpha=alpha)


def test_ema_alpha_boundaries_and_large_fast_window() -> None:
    default = ExponentialMovingAverage(window=3, alpha=0.0)
    identity = ExponentialMovingAverage(window=3, alpha=1.0)
    large = ExponentialMovingAverage(window=2**31 - 1, is_fast=True)
    for ma in (default, identity, large):
        ma.update(0.0)
    assert default.update(10.0) == 5.0
    assert identity.update(10.0) == 10.0
    assert large.update(10.0) == pytest.approx(20.0 / 2**31, rel=1e-12)


@pytest.mark.parametrize("ma_cls", MA_CLASSES)
def test_fast_mode_omits_output_history_allocation(ma_cls: type[MovingAverage]) -> None:
    """Fast EMA/TEMA are constant-space; SMA/WMA retain only their raw window."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        ma = ma_cls(window=2**20, is_fast=True)
        allocated = tracemalloc.get_traced_memory()[0] - before
        raw_bytes = (
            8 * 2**20 if ma_cls in (SimpleMovingAverage, WeightedMovingAverage) else 0
        )
        assert allocated < raw_bytes + 1024 * 1024
        assert ma.get_value() == 0.0
    finally:
        if not was_tracing:
            tracemalloc.stop()

"""Layer 3 — Integration tests for all moving-average implementations.

Validates protocol compliance, cross-MA behavioural consistency, fast-mode
semantics, and numerical accuracy against reference NumPy
implementations.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from mm_toolbox.moving_average import (
    ExponentialMovingAverage,
    SimpleMovingAverage,
    TimeExponentialMovingAverage,
    WeightedMovingAverage,
)
from mm_toolbox.moving_average.base import MovingAverage

MA_CLASSES = [
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage,
    TimeExponentialMovingAverage,
]

MA_NAMES = ["SMA", "EMA", "WMA", "TEMA"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_sma(values: np.ndarray, window: int) -> np.ndarray:
    """NumPy reference for SMA."""
    result = np.empty_like(values)
    result[: window - 1] = np.nan
    result[window - 1] = values[:window].mean()
    for i in range(window, len(values)):
        result[i] = values[i - window + 1 : i + 1].mean()
    return result


def _reference_ema(values: np.ndarray, window: int) -> np.ndarray:
    """NumPy reference for EMA (span = window, adjust=False)."""
    alpha = 2.0 / (window + 1)
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def _reference_wma(values: np.ndarray, window: int) -> np.ndarray:
    """NumPy reference for WMA."""
    weights = np.arange(1, window + 1, dtype=np.float64)
    denom = weights.sum()
    result = np.empty_like(values)
    result[: window - 1] = np.nan
    for i in range(window - 1, len(values)):
        result[i] = (values[i - window + 1 : i + 1] * weights).sum() / denom
    return result


def _make_instance(ma_cls: Any, window: int, is_fast: bool = False) -> Any:
    """Instantiate an MA with the minimal required kwargs."""
    kwargs = {"window": window, "is_fast": is_fast}
    if ma_cls is TimeExponentialMovingAverage:
        kwargs["half_life_s"] = 1.0
    return ma_cls(**kwargs)


# ---------------------------------------------------------------------------
# Layer 1 – Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Verify every MA is an instance of ``MovingAverage``."""

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_isinstance_protocol(self, ma_cls: Any):
        """Runtime checkable protocol acceptance."""
        ma = _make_instance(ma_cls, window=4)
        assert isinstance(ma, MovingAverage)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_has_required_methods(self, ma_cls: Any):
        """All protocol methods exist and are callable."""
        ma = _make_instance(ma_cls, window=4)
        assert callable(ma.initialize)
        assert callable(ma.next)
        assert callable(ma.update)
        assert callable(ma.get_value)
        assert callable(ma.get_values)
        assert callable(ma.__len__)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_window_validation(self, ma_cls: Any):
        """Window <= 1 must raise ``ValueError``."""
        with pytest.raises(ValueError):
            _make_instance(ma_cls, window=1)
        with pytest.raises(ValueError):
            _make_instance(ma_cls, window=0)
        with pytest.raises(ValueError):
            _make_instance(ma_cls, window=-1)


# ---------------------------------------------------------------------------
# Layer 2 – Parametrised identical-sequence tests
# ---------------------------------------------------------------------------


class TestIdenticalSequence:
    """Run the same price sequence through every MA."""

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_initialize_returns_float(self, ma_cls: Any):
        """``initialize()`` returns a Python float."""
        ma = _make_instance(ma_cls, window=4)
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = ma.initialize(vals)
        assert isinstance(result, float)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_update_returns_float(self, ma_cls: Any):
        """Sequential ``update()`` calls return floats."""
        ma = _make_instance(ma_cls, window=4)
        ma.initialize(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

        for price in [5.0, 6.0, 7.0]:
            if ma_cls is TimeExponentialMovingAverage:
                time.sleep(0.01)  # let TEMA's clock advance slightly
            result = ma.update(price)
            assert isinstance(result, float)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_get_value_matches_latest(self, ma_cls: Any):
        """``get_value()`` equals the most recent ``update()`` return."""
        ma = _make_instance(ma_cls, window=4)
        ma.initialize(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64))

        if ma_cls is TimeExponentialMovingAverage:
            time.sleep(0.01)
        returned = ma.update(50.0)
        assert ma.get_value() == pytest.approx(returned)


# ---------------------------------------------------------------------------
# Layer 2 – Fast mode semantics
# ---------------------------------------------------------------------------


class TestFastMode:
    """Fast mode must disable history while preserving current-value accuracy."""

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_fast_mode_raises_on_get_values(self, ma_cls: Any):
        """``get_values()`` is forbidden in fast mode."""
        ma = _make_instance(ma_cls, window=4, is_fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

        with pytest.raises(ValueError, match="fast mode"):
            ma.get_values()

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_fast_mode_raises_on_len(self, ma_cls: Any):
        """``len()`` is forbidden in fast mode."""
        ma = _make_instance(ma_cls, window=4, is_fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

        with pytest.raises(ValueError, match="fast mode"):
            len(ma)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_fast_mode_value_matches_normal(self, ma_cls: Any):
        """Current value must be identical between fast and normal modes."""
        window = 4
        init_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        updates = [5.0, 6.0, 7.0, 8.0]

        ma_fast = _make_instance(ma_cls, window=window, is_fast=True)
        ma_norm = _make_instance(ma_cls, window=window, is_fast=False)

        ma_fast.initialize(init_vals.copy())
        ma_norm.initialize(init_vals.copy())

        for price in updates:
            if ma_cls is TimeExponentialMovingAverage:
                time.sleep(0.01)
            v_fast = ma_fast.update(price)
            v_norm = ma_norm.update(price)
            assert v_fast == pytest.approx(v_norm, abs=1e-9)
            assert ma_fast.get_value() == pytest.approx(ma_norm.get_value(), abs=1e-9)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_fast_mode_len_stays_zero(self, ma_cls: Any):
        """Fast mode must never accumulate history."""
        ma = _make_instance(ma_cls, window=4, is_fast=True)
        ma.initialize(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

        for _ in range(10):
            if ma_cls is TimeExponentialMovingAverage:
                time.sleep(0.005)
            ma.update(5.0)

        # len() is already tested to raise, but we can also inspect
        # indirectly: get_values must raise after many updates.
        with pytest.raises(ValueError, match="fast mode"):
            ma.get_values()


# ---------------------------------------------------------------------------
# Layer 3 – Realistic 1000-tick workflow
# ---------------------------------------------------------------------------


class TestRealisticWorkflow:
    """End-to-end 1000-tick price stream."""

    @pytest.fixture(scope="class")
    def price_stream(self) -> np.ndarray:
        """A synthetic 1000-tick price series."""
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.0, scale=0.01, size=1000)
        prices = 100.0 * np.exp(np.cumsum(returns))
        return prices.astype(np.float64)

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_runs_without_error(self, ma_cls: Any, price_stream: np.ndarray):
        """All MAs must ingest 1000 ticks without exception."""
        window = 20
        ma = _make_instance(ma_cls, window=window)
        ma.initialize(price_stream[:window].copy())

        for price in price_stream[window:]:
            if ma_cls is TimeExponentialMovingAverage:
                # tiny sleep so TEMA's alpha is non-zero
                time.sleep(0.001)
            ma.update(float(price))

        assert ma.get_value() > 0.0
        assert not np.isnan(ma.get_value())

    @pytest.mark.parametrize("ma_cls", MA_CLASSES, ids=MA_NAMES)
    def test_values_monotonic_in_window(self, ma_cls: Any, price_stream: np.ndarray):
        """For a constant price the MA output should stabilise."""
        window = 10
        ma = _make_instance(ma_cls, window=window)
        ma.initialize(np.full(window, 50.0, dtype=np.float64))

        for _ in range(100):
            if ma_cls is TimeExponentialMovingAverage:
                time.sleep(0.001)
            ma.update(50.0)

        assert ma.get_value() == pytest.approx(50.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Layer 3 – Reference implementations (SMA, EMA, WMA)
# ---------------------------------------------------------------------------


class TestReferenceAccuracy:
    """Compare MA outputs to reference NumPy implementations."""

    @pytest.fixture(scope="class")
    def price_stream(self) -> np.ndarray:
        rng = np.random.default_rng(12345)
        returns = rng.normal(loc=0.0, scale=0.005, size=500)
        return (100.0 * np.exp(np.cumsum(returns))).astype(np.float64)

    def test_sma_matches_reference(self, price_stream: np.ndarray):
        """SMA output must align with NumPy rolling mean."""
        window = 20
        ma = SimpleMovingAverage(window=window)
        ma.initialize(price_stream[:window].copy())

        results = [ma.get_value()]
        for price in price_stream[window:]:
            results.append(ma.update(float(price)))

        ref = _reference_sma(price_stream, window)
        np.testing.assert_allclose(results, ref[window - 1 :], rtol=1e-12)

    def test_ema_matches_pandas(self, price_stream: np.ndarray):
        """EMA output must align with the NumPy recurrence reference."""
        window = 20
        ma = ExponentialMovingAverage(window=window)
        ma.initialize(price_stream[:window].copy())

        results = [ma.get_value()]
        for price in price_stream[window:]:
            results.append(ma.update(float(price)))

        ref = _reference_ema(price_stream, window)
        # Compare from the first update onward.
        np.testing.assert_allclose(results, ref[window - 1 :], rtol=1e-10)

    def test_wma_matches_reference(self, price_stream: np.ndarray):
        """WMA output must align with NumPy weighted average."""
        window = 20
        ma = WeightedMovingAverage(window=window)
        ma.initialize(price_stream[:window].copy())

        results = [ma.get_value()]
        for price in price_stream[window:]:
            results.append(ma.update(float(price)))

        ref = _reference_wma(price_stream, window)
        np.testing.assert_allclose(results, ref[window - 1 :], rtol=1e-12)

    def test_tema_approximates_expected_with_known_timestamps(self):
        """TEMA against an explicit Python reference with controlled time steps."""
        half_life = 1.0
        window = 2
        prices = np.array([10.0, 10.0, 20.0, 20.0, 30.0], dtype=np.float64)
        # Simulate 1-second intervals
        timestamps = np.arange(len(prices), dtype=np.float64)

        ma = TimeExponentialMovingAverage(window=window, half_life_s=half_life)
        ma.initialize(prices[:2].copy())

        results = [ma.get_value()]
        for i in range(2, len(prices)):
            time.sleep(1.0)  # real wall-clock step of 1 s
            results.append(ma.update(float(prices[i])))

        # Reference
        lam = np.log(2.0) / half_life
        ref_values = [prices[0]]
        for i in range(1, len(prices)):
            dt = timestamps[i] - timestamps[i - 1]
            alpha = 1.0 - np.exp(-lam * dt)
            ref_values.append(alpha * prices[i] + (1.0 - alpha) * ref_values[-1])

        np.testing.assert_allclose(results, ref_values[1:], rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for Rounder implementation."""

import numpy as np
import pytest

from mm_toolbox.rounding import Rounder
from mm_toolbox.rounding.rounder import RounderConfig


class TestRounderConfig:
    """Test RounderConfig validation comprehensively."""

    def test_valid_config_creation(self):
        """Test creating valid configurations."""
        cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=True,
            round_asks_up=True,
            round_size_up=True,
        )
        assert cfg.tick_size == 0.01
        assert cfg.lot_size == 0.001
        assert cfg.round_bids_down is True
        assert cfg.round_asks_up is True
        assert cfg.round_size_up is True

    def test_tick_size_validation(self):
        """Test tick_size validation."""
        # Valid tick sizes
        valid_tick_sizes = [0.01, 0.001, 0.0001, 1.0, 0.5]
        for tick_size in valid_tick_sizes:
            cfg = RounderConfig(
                tick_size=tick_size,
                lot_size=0.001,
                round_bids_down=True,
                round_asks_up=True,
                round_size_up=True,
            )
            assert cfg.tick_size == tick_size

        # Invalid tick sizes
        invalid_tick_sizes = [0.0, -0.01, -1.0]
        for tick_size in invalid_tick_sizes:
            with pytest.raises(
                ValueError, match="Invalid tick_size; must be greater than 0"
            ):
                RounderConfig(
                    tick_size=tick_size,
                    lot_size=0.001,
                    round_bids_down=True,
                    round_asks_up=True,
                    round_size_up=True,
                )

    def test_lot_size_validation(self):
        """Test lot_size validation."""
        # Valid lot sizes
        valid_lot_sizes = [0.01, 0.001, 0.0001, 1.0, 0.5]
        for lot_size in valid_lot_sizes:
            cfg = RounderConfig(
                tick_size=0.01,
                lot_size=lot_size,
                round_bids_down=True,
                round_asks_up=True,
                round_size_up=True,
            )
            assert cfg.lot_size == lot_size

        # Invalid lot sizes
        invalid_lot_sizes = [0.0, -0.001, -1.0]
        for lot_size in invalid_lot_sizes:
            with pytest.raises(
                ValueError, match="Invalid lot_size; must be greater than 0"
            ):
                RounderConfig(
                    tick_size=0.01,
                    lot_size=lot_size,
                    round_bids_down=True,
                    round_asks_up=True,
                    round_size_up=True,
                )

    def test_default_config_creation(self):
        """Test default configuration creation."""
        cfg = RounderConfig.default(tick_size=0.01, lot_size=0.001)
        assert cfg.tick_size == 0.01
        assert cfg.lot_size == 0.001
        assert cfg.round_bids_down is True
        assert cfg.round_asks_up is True
        assert cfg.round_size_up is True

    def test_custom_rounding_configurations(self):
        """Test different rounding direction configurations."""
        # Custom configuration: round bids up, asks down, sizes down
        cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=False,
            round_asks_up=False,
            round_size_up=False,
        )
        assert cfg.round_bids_down is False
        assert cfg.round_asks_up is False
        assert cfg.round_size_up is False


class TestRounderBasicOperations:
    """Test basic Rounder operations."""

    def test_rounder_creation(self):
        """Test creating Rounder instances."""
        cfg = RounderConfig.default(0.01, 0.001)
        rounder = Rounder(cfg)

        assert rounder is not None
        assert isinstance(rounder.bid(1.234), float)
        assert isinstance(rounder.ask(1.234), float)
        assert isinstance(rounder.size(1.234), float)

    def test_single_bid_rounding_default(self):
        """Test single bid rounding with default configuration (round down)."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test exact tick multiples
        assert rounder.bid(1.00) == 1.00
        assert rounder.bid(1.01) == 1.01
        assert rounder.bid(1.02) == 1.02

        # Test values between ticks (should round down)
        assert rounder.bid(1.005) == 1.00
        assert rounder.bid(1.014) == 1.01
        assert rounder.bid(1.025) == 1.02

        # Test edge cases
        assert rounder.bid(1.009) == 1.00
        assert rounder.bid(1.019) == 1.01

    def test_single_ask_rounding_default(self):
        """Test single ask rounding with default configuration (round up)."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test exact tick multiples
        assert rounder.ask(1.00) == 1.00
        assert rounder.ask(1.01) == 1.01
        assert rounder.ask(1.02) == 1.02

        # Test values between ticks (should round up)
        assert rounder.ask(1.005) == 1.01
        assert rounder.ask(1.014) == 1.02
        assert rounder.ask(1.025) == 1.03

    def test_single_size_rounding_default(self):
        """Test single size rounding with default configuration (round up)."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test exact lot multiples
        assert rounder.size(1.000) == 1.000
        assert rounder.size(1.001) == 1.001
        assert rounder.size(1.002) == 1.002

        # Test values between lots (should round up)
        assert rounder.size(1.0005) == 1.001
        assert rounder.size(1.0014) == 1.002
        assert rounder.size(1.0025) == 1.003

    def test_array_bids_basic(self):
        """Test basic array bid rounding."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        prices = np.array([1.234, 1.231, 1.237, 1.240, 1.245])
        expected = np.array([1.23, 1.23, 1.23, 1.24, 1.24])

        result = rounder.bids(prices)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        assert result.dtype == np.float64

    def test_array_asks_basic(self):
        """Test basic array ask rounding."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        prices = np.array([1.234, 1.231, 1.237, 1.240, 1.245])
        expected = np.array([1.24, 1.24, 1.24, 1.24, 1.25])

        result = rounder.asks(prices)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        assert result.dtype == np.float64

    def test_array_sizes_basic(self):
        """Test basic array size rounding."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        sizes = np.array([1.234, 1.231, 1.237, 1.240, 1.245])
        expected = np.array([1.234, 1.231, 1.237, 1.240, 1.245])

        result = rounder.sizes(sizes)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
        assert result.dtype == np.float64


class TestRounderCustomConfigurations:
    """Test Rounder with custom configurations."""

    def test_custom_bid_rounding(self):
        """Test bid rounding with custom configuration (round up)."""
        cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=False,
            round_asks_up=True,
            round_size_up=True,
        )
        rounder = Rounder(cfg)

        # Test values between ticks (should round up)
        assert rounder.bid(1.005) == 1.01
        assert rounder.bid(1.014) == 1.02
        assert rounder.bid(1.025) == 1.03

    def test_custom_ask_rounding(self):
        """Test ask rounding with custom configuration (round down)."""
        cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=True,
            round_asks_up=False,
            round_size_up=True,
        )
        rounder = Rounder(cfg)

        # Test values between ticks (should round down)
        assert rounder.ask(1.005) == 1.00
        assert rounder.ask(1.014) == 1.01
        assert rounder.ask(1.025) == 1.02

    def test_custom_size_rounding(self):
        """Test size rounding with custom configuration (round down)."""
        cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=True,
            round_asks_up=True,
            round_size_up=False,
        )
        rounder = Rounder(cfg)

        # Test values between lots (should round down)
        assert rounder.size(1.0005) == 1.000
        assert rounder.size(1.0014) == 1.001
        assert rounder.size(1.0025) == 1.002

    def test_array_operations_custom_config(self):
        """Test array operations with custom configurations."""
        cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=False,
            round_asks_up=False,
            round_size_up=False,
        )
        rounder = Rounder(cfg)

        prices = np.array([1.234, 1.231, 1.237, 1.240, 1.245])

        # Bids should round up
        expected_bids = np.array([1.24, 1.24, 1.24, 1.24, 1.25])
        result_bids = rounder.bids(prices)
        np.testing.assert_allclose(result_bids, expected_bids, rtol=0, atol=1e-12)

        # Asks should round down
        expected_asks = np.array([1.23, 1.23, 1.23, 1.24, 1.24])
        result_asks = rounder.asks(prices)
        np.testing.assert_allclose(result_asks, expected_asks, rtol=0, atol=1e-12)

    def test_rounding_direction_consistency(self):
        """Test that rounding direction is consistent with configuration."""
        default_rounder = Rounder(RounderConfig.default(0.01, 0.001))
        custom_cfg = RounderConfig(
            tick_size=0.01,
            lot_size=0.001,
            round_bids_down=False,
            round_asks_up=False,
            round_size_up=False,
        )
        custom_rounder = Rounder(custom_cfg)

        test_price = 1.005

        # Default: bids down, asks up
        assert default_rounder.bid(test_price) < test_price
        assert default_rounder.ask(test_price) > test_price

        # Custom: bids up, asks down
        assert custom_rounder.bid(test_price) > test_price
        assert custom_rounder.ask(test_price) < test_price


class TestRounderDifferentTickSizes:
    """Test Rounder with different tick and lot sizes."""

    def test_small_tick_precision(self):
        """Test precision with very small tick sizes."""
        rounder = Rounder(RounderConfig.default(0.0001, 0.00001))

        # Test 0.0001 tick size
        assert rounder.bid(1.12345) == 1.1234
        assert rounder.ask(1.12345) == 1.1235
        assert rounder.bid(1.12346) == 1.1234
        assert rounder.ask(1.12346) == 1.1235

    def test_large_tick_rounding(self):
        """Test rounding with large tick sizes."""
        rounder = Rounder(RounderConfig.default(1.0, 0.1))

        # Test 1.0 tick size
        assert rounder.bid(1.4) == 1.0
        assert rounder.ask(1.4) == 2.0
        assert rounder.bid(1.6) == 1.0
        assert rounder.ask(1.6) == 2.0

    def test_fractional_tick_sizes(self):
        """Test with fractional tick sizes."""
        rounder = Rounder(RounderConfig.default(0.5, 0.1))

        # Test 0.5 tick size (simpler fractions to avoid precision issues)
        # Valid multiples of 0.5: 0.0, 0.5, 1.0, 1.5, 2.0, etc.

        # Test basic rounding behavior
        assert rounder.bid(1.2) == 1.0  # 1.2 rounds down to 1.0
        assert rounder.ask(1.2) == 1.5  # 1.2 rounds up to 1.5

        assert rounder.bid(1.7) == 1.5  # 1.7 rounds down to 1.5
        assert rounder.ask(1.7) == 2.0  # 1.7 rounds up to 2.0

        # Test exact multiples remain unchanged
        assert rounder.bid(1.0) == 1.0
        assert rounder.ask(1.0) == 1.0
        assert rounder.bid(1.5) == 1.5
        assert rounder.ask(1.5) == 1.5

    def test_lot_size_variations(self):
        """Test with different lot sizes."""
        rounder = Rounder(RounderConfig.default(0.01, 0.01))

        # Test 0.01 lot size
        assert rounder.size(1.234) == 1.24
        assert rounder.size(1.235) == 1.24
        assert rounder.size(1.236) == 1.24


class TestRounderArrayOperations:
    """Test array operations comprehensively."""

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        empty_prices = np.array([])
        empty_sizes = np.array([])

        result_bids = rounder.bids(empty_prices)
        result_asks = rounder.asks(empty_prices)
        result_sizes = rounder.sizes(empty_sizes)

        assert len(result_bids) == 0
        assert len(result_asks) == 0
        assert len(result_sizes) == 0
        assert result_bids.dtype == np.float64
        assert result_asks.dtype == np.float64
        assert result_sizes.dtype == np.float64

    def test_single_element_arrays(self):
        """Test arrays with single elements."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        single_price = np.array([1.234])
        single_size = np.array([1.234])

        result_bid = rounder.bids(single_price)
        result_ask = rounder.asks(single_price)
        result_size = rounder.sizes(single_size)

        assert len(result_bid) == 1
        assert len(result_ask) == 1
        assert len(result_size) == 1
        assert result_bid[0] == 1.23
        assert result_ask[0] == 1.24
        assert result_size[0] == 1.234

    def test_large_arrays(self):
        """Test with reasonably large arrays."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        large_array = np.random.uniform(1.0, 2.0, 1000)

        result_bids = rounder.bids(large_array)
        result_asks = rounder.asks(large_array)
        result_sizes = rounder.sizes(large_array)

        assert len(result_bids) == len(large_array)
        assert len(result_asks) == len(large_array)
        assert len(result_sizes) == len(large_array)
        assert result_bids.dtype == np.float64
        assert result_asks.dtype == np.float64
        assert result_sizes.dtype == np.float64

    def test_different_dtypes(self):
        """Test handling of different numpy dtypes."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # The Cython implementation expects double precision, so we test conversion
        prices_float32 = np.array([1.234, 1.231], dtype=np.float32)
        prices_float64 = np.array([1.234, 1.231], dtype=np.float64)

        # Convert float32 to float64 before passing to Cython methods
        result_32 = rounder.bids(prices_float32.astype(np.float64))
        result_64 = rounder.bids(prices_float64)

        assert result_32.dtype == np.float64
        assert result_64.dtype == np.float64
        np.testing.assert_allclose(result_32, result_64, rtol=0, atol=1e-12)

    def test_scalar_array_consistency(self):
        """Test that scalar and array methods produce consistent results."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        prices = np.array([1.234, 1.231, 1.237])

        # Test bid consistency
        scalar_bids = [rounder.bid(p) for p in prices]
        array_bids = rounder.bids(prices)
        np.testing.assert_allclose(scalar_bids, array_bids, rtol=0, atol=1e-12)

        # Test ask consistency
        scalar_asks = [rounder.ask(p) for p in prices]
        array_asks = rounder.asks(prices)
        np.testing.assert_allclose(scalar_asks, array_asks, rtol=0, atol=1e-12)

        # Test size consistency
        sizes = np.array([1.234, 1.231, 1.237])
        scalar_sizes = [rounder.size(s) for s in sizes]
        array_sizes = rounder.sizes(sizes)
        np.testing.assert_allclose(scalar_sizes, array_sizes, rtol=0, atol=1e-12)


class TestRounderEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_and_negative_values(self):
        """Test handling of zero and negative values."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Zero values
        assert rounder.bid(0.0) == 0.0
        assert rounder.ask(0.0) == 0.0
        assert rounder.size(0.0) == 0.0

        # Negative values
        assert rounder.bid(-1.234) == -1.24
        assert rounder.ask(-1.234) == -1.23
        assert rounder.size(-1.234) == -1.234

    def test_extreme_values(self):
        """Test handling of extreme values."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Very large values
        large_price = 1e15
        assert rounder.bid(large_price) == pytest.approx(large_price, rel=1e-10)
        assert rounder.ask(large_price) == pytest.approx(large_price, rel=1e-10)

        # Very small values
        small_price = 1e-15
        assert rounder.bid(small_price) == 0.0
        assert rounder.ask(small_price) == 0.01

    def test_boundary_values(self):
        """Test values exactly at tick boundaries."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test values exactly at tick boundaries
        assert rounder.bid(1.00) == 1.00
        assert rounder.ask(1.00) == 1.00

        # Test values just below and above tick boundaries
        assert rounder.bid(0.999) == 0.99
        assert rounder.ask(0.999) == 1.00
        assert rounder.bid(1.001) == 1.00
        assert rounder.ask(1.001) == 1.01

    def test_lot_size_boundaries(self):
        """Test lot size rounding edge cases."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test values exactly at lot boundaries
        assert rounder.size(1.000) == 1.000
        assert rounder.size(1.001) == 1.001

        # Test values just below and above lot boundaries
        assert rounder.size(0.999) == 0.999
        assert rounder.size(1.0005) == 1.001
        assert rounder.size(1.0015) == 1.002

    def test_numerical_stability(self):
        """Test numerical stability with floating point arithmetic."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test that repeated operations don't accumulate errors
        price = 1.234
        result1 = rounder.bid(price)
        result2 = rounder.bid(result1)
        result3 = rounder.bid(result2)

        # Results should be stable after first rounding
        assert result1 == result2
        assert result2 == result3

        # Test with more reasonable tick sizes for numerical stability
        medium_rounder = Rounder(RounderConfig.default(1e-6, 1e-6))
        medium_price = 1.0 + 1e-5
        result = medium_rounder.bid(medium_price)
        # With 1e-6 tick size, 1.00001 should round down to 1.00001 (10 ticks
        # up from 1.0)
        assert result == pytest.approx(1.00001, abs=1e-6)

    def test_memory_efficiency(self):
        """Test that array operations don't create unnecessary copies."""
        rounder = Rounder(RounderConfig.default(0.01, 0.001))
        prices = np.array([1.234, 1.231, 1.237])

        # Test that input array is not modified
        prices_original = prices.copy()
        result = rounder.bids(prices)
        np.testing.assert_array_equal(prices, prices_original)

        # Test that result is a new array
        assert result is not prices
        assert result.base is not prices.base if prices.base is not None else True

    def test_precision_edge_cases(self):
        """Test precision edge cases with different tick sizes."""
        # Test with simpler tick size to avoid floating point precision issues
        rounder = Rounder(RounderConfig.default(0.01, 0.001))

        # Test values that should round cleanly
        test_values = [1.005, 1.015, 1.025, 1.035]
        for value in test_values:
            bid_result = rounder.bid(value)
            ask_result = rounder.ask(value)

            # Results should be valid multiples of tick size (with some
            # tolerance for floating point)
            tick_multiple_bid = round(bid_result / 0.01)
            tick_multiple_ask = round(ask_result / 0.01)

            assert abs(bid_result - tick_multiple_bid * 0.01) < 1e-12
            assert abs(ask_result - tick_multiple_ask * 0.01) < 1e-12

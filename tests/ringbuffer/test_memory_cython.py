"""Wrapper to expose layered native Cython tests for memory utilities to pytest.

Tests primitive memory operations: align_up, pow2_at_least, write_u64_le,
read_u64_le, copy_into_ring, and copy_from_ring. These are Layer 1–3 tests
executed via the compiled Cython test module.
"""

from __future__ import annotations

import os
import sys

import pytest

# Go up to tests/ directory to find the compiled .so files
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

try:
    import cython_test_memory as _native
except ImportError as e:
    pytest.skip(f"Native Cython test module not built: {e}", allow_module_level=True)


# =============================================================================
# Layer 1: Primitives - align_up and pow2_at_least
# =============================================================================


class TestAlignUp:
    """Layer 1 tests for align_up function."""

    def test_align_up_zero(self):
        """Given x=0, When aligned up, Then result is 0."""
        _native.test_align_up_zero()

    def test_align_up_less_than_align(self):
        """Given x < a, When aligned up, Then result is a."""
        _native.test_align_up_less_than_align()

    def test_align_up_equals_align(self):
        """Given x == a, When aligned up, Then result is x."""
        _native.test_align_up_equals_align()

    def test_align_up_greater_than_align(self):
        """Given x > a, When aligned up, Then result is next multiple."""
        _native.test_align_up_greater_than_align()

    def test_align_up_align_one(self):
        """Given a=1, When aligned up, Then result equals x."""
        _native.test_align_up_align_one()


class TestPow2AtLeast:
    """Layer 1 tests for pow2_at_least function."""

    def test_pow2_at_least_zero(self):
        """Given v=0, When computing pow2_at_least, Then result is 1."""
        _native.test_pow2_at_least_zero()

    def test_pow2_at_least_one(self):
        """Given v=1, When computing pow2_at_least, Then result is 1."""
        _native.test_pow2_at_least_one()

    def test_pow2_at_least_two(self):
        """Given v=2, When computing pow2_at_least, Then result is 2."""
        _native.test_pow2_at_least_two()

    def test_pow2_at_least_three(self):
        """Given v=3, When computing pow2_at_least, Then result is 4."""
        _native.test_pow2_at_least_three()

    def test_pow2_at_least_already_power_of_two(self):
        """Given v is already a power of two, Then result is v."""
        _native.test_pow2_at_least_already_power_of_two()

    def test_pow2_at_least_large_value(self):
        """Given v=UINT64_MAX/2, When computing pow2_at_least, Then result is 2^63."""
        _native.test_pow2_at_least_large_value()

    def test_pow2_at_least_just_above_power_of_two(self):
        """Given v just above a power of two, Then result is next power of two."""
        _native.test_pow2_at_least_just_above_power_of_two()


# =============================================================================
# Layer 2: Composite Components - write_u64_le and read_u64_le
# =============================================================================


class TestWriteReadU64Le:
    """Layer 2 tests for write_u64_le and read_u64_le functions."""

    def test_write_read_u64_le_basic(self):
        """Given aligned position, When writing and reading u64, Then value matches."""
        _native.test_write_read_u64_le_basic()

    def test_write_read_u64_le_different_values(self):
        """Given various u64 values, When written and read back, Then values match."""
        _native.test_write_read_u64_le_different_values()

    def test_write_read_u64_le_wraparound(self):
        """Given position near end of buffer, When writing 8 bytes, Then wrap path used."""
        _native.test_write_read_u64_le_wraparound()

    def test_write_read_u64_le_boundary_capacity_minus_8(self):
        """Given position at capacity-8, When writing 8 bytes, Then fast path used."""
        _native.test_write_read_u64_le_boundary_capacity_minus_8()

    def test_write_read_u64_le_multiple_wrap_positions(self):
        """Given various wrap positions, When writing and reading, Then all values match."""
        _native.test_write_read_u64_le_multiple_wrap_positions()


# =============================================================================
# Layer 3: Mini-Integration - copy_into_ring and copy_from_ring
# =============================================================================


class TestCopyIntoRing:
    """Layer 3 tests for copy_into_ring function."""

    def test_copy_into_ring_no_wrap(self):
        """Given n < endspace, When copying into ring, Then data intact."""
        _native.test_copy_into_ring_no_wrap()

    def test_copy_into_ring_exact_endspace(self):
        """Given n == endspace, When copying into ring, Then data intact."""
        _native.test_copy_into_ring_exact_endspace()

    def test_copy_into_ring_wrap(self):
        """Given n > endspace, When copying into ring, Then wrap-around used."""
        _native.test_copy_into_ring_wrap()

    def test_copy_into_ring_zero(self):
        """Given n=0, When copying into ring, Then no change."""
        _native.test_copy_into_ring_zero()


class TestCopyFromRing:
    """Layer 3 tests for copy_from_ring function."""

    def test_copy_from_ring_no_wrap(self):
        """Given n < endspace, When copying from ring, Then data matches."""
        _native.test_copy_from_ring_no_wrap()

    def test_copy_from_ring_exact_endspace(self):
        """Given n == endspace, When copying from ring, Then data matches."""
        _native.test_copy_from_ring_exact_endspace()

    def test_copy_from_ring_wrap(self):
        """Given n > endspace, When copying from ring, Then wrap-around used."""
        _native.test_copy_from_ring_wrap()

    def test_copy_from_ring_zero(self):
        """Given n=0, When copying from ring, Then destination unchanged."""
        _native.test_copy_from_ring_zero()


class TestCopyRoundtrip:
    """Layer 3 integration test for copy roundtrip."""

    def test_copy_roundtrip(self):
        """Given data copied into ring, When copied back out, Then data matches."""
        _native.test_copy_roundtrip()

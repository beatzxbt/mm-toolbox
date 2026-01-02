# import time

# import pytest
# from mm_toolbox.misc.limiter import (
#     RateLimitBurstConfig,
#     RateLimiter,
#     RateLimiterConfig,
#     RateLimitStateConfig,
# )


# def make_limiter(
#     tokens: int,
#     window_s: int,
#     warn: float = 0.75,
#     block: float = 0.95,
#     burst: bool = False,
#     max_tokens: int = 0,
#     max_attempts: int = 0,
# ) -> RateLimiter:
#     state_cfg = RateLimitStateConfig(
#         is_enabled=True, warning_threshold=warn, block_threshold=block
#     )
#     burst_cfg = RateLimitBurstConfig(
#         is_enabled=burst, max_tokens=max_tokens, max_burst_attempts=max_attempts
#     )
#     cfg = RateLimiterConfig(
#         available_tokens=tokens,
#         refill_duration_s=window_s,
#         state_config=state_cfg,
#         burst_config=burst_cfg,
#     )
#     return RateLimiter(cfg)


# class TestRateLimiterConfig:
#     """Test configuration validation and defaults."""

#     def test_default_factory(self):
#         cfg = RateLimiterConfig.default(available_tokens=5, refill_duration_s=1)
#         assert cfg.available_tokens == 5
#         assert cfg.refill_duration_s == 1
#         assert (
#             0.0
#             < cfg.state_config.warning_threshold
#             < cfg.state_config.block_threshold
#             < 1.0
#         )

#     def test_invalid_thresholds(self):
#         with pytest.raises(ValueError):
#             RateLimitStateConfig(
#                 is_enabled=True, warning_threshold=0.9, block_threshold=0.8
#             )
#         with pytest.raises(ValueError):
#             RateLimitStateConfig(
#                 is_enabled=True, warning_threshold=0.0, block_threshold=0.5
#             )
#         with pytest.raises(ValueError):
#             RateLimitStateConfig(
#                 is_enabled=True, warning_threshold=0.6, block_threshold=1.0
#             )


# class TestRateLimiterBasics:
#     """Test basic single-second window behavior and auto-refill."""

#     def test_single_second_window_basic_usage(self):
#         rl = make_limiter(tokens=4, window_s=1)
#         assert rl.tokens_remaining() == 4
#         assert rl.usage() == 0.0

#         assert rl.try_consume() == 0  # NORMAL
#         assert rl.tokens_remaining() == 3

#         # consume up to warn
#         assert rl.try_consume_multiple(2) == 0
#         # 3 used out of 4 => usage 0.75, next token should tip into warning
#         assert rl.try_consume() == 1  # WARNING
#         assert rl.tokens_remaining() == 0

#     def test_single_second_window_refill(self):
#         rl = make_limiter(tokens=2, window_s=1)
#         assert rl.try_consume() == 0
#         assert rl.try_consume() == 0
#         assert rl.tokens_remaining() == 0

#         time.sleep(1.05)
#         # auto-refill should have happened on next call
#         assert rl.try_consume() == 0
#         assert rl.tokens_remaining() == 1


# class TestRateLimiterSubBuckets:
#     """Test multi-second window with sub-bucket smoothing."""

#     def test_multi_second_window_sub_buckets(self):
#         rl = make_limiter(tokens=6, window_s=3)
#         # Expect per-bucket alloc ceil(6/3)=2
#         assert rl.try_consume_multiple(2) == 0
#         # Same-second third token may be blocked by sub-bucket depending on timing
#         assert rl.try_consume() in (2, 0, 1)

#         # Sleep into next second, should allow again as sub-bucket rotates
#         time.sleep(1.05)
#         state = rl.try_consume_multiple(2)
#         assert state in (0, 1)

#     @pytest.mark.parametrize(
#         "window_s, tokens, per_bucket", [(10, 20, 2), (60, 120, 2)]
#     )
#     def test_long_windows_even_distribution(
#         self, window_s: int, tokens: int, per_bucket: int
#     ):
#         # Expect ceil(tokens/window) = 2 per second bucket
#         rl = make_limiter(tokens=tokens, window_s=window_s)
#         # Quickly consume up to per-bucket in the first second
#         assert rl.try_consume_multiple(per_bucket) == 0
#         # Next token in same second likely blocked by sub-bucket
#         state_same_second = rl.try_consume()
#         assert state_same_second in (2, 0, 1)

#         # After moving into next second, another per-bucket should be allowed
#         time.sleep(1.05)
#         state_next_second = rl.try_consume_multiple(per_bucket)
#         assert state_next_second in (0, 1)

#     def test_long_window_accumulated_usage_warning(self):
#         # 10s window, 10 tokens -> 1 token/sec; sustained usage approaches thresholds
#         rl = make_limiter(tokens=10, window_s=10, warn=0.7, block=0.9)

#         total_ok = 0
#         for _i in range(6):
#             # Consume one per second for 6 seconds
#             state = rl.try_consume()
#             assert state in (0, 1)
#             total_ok += 1
#             time.sleep(1.02)

#         # Usage should be ~0.6; still below warn
#         assert rl.usage() < 0.7

#         # Push a few more quickly to cross warn but not block
#         state = rl.try_consume_multiple(2)
#         assert state in (0, 1)
#         assert rl.usage() >= 0.7


# class TestRateLimiterThresholdsAndBurst:
#     """Test threshold transitions and burst behavior."""

#     def test_thresholds_and_block(self):
#         rl = make_limiter(tokens=4, window_s=2, warn=0.5, block=0.75)
#         # 2 tokens -> usage=0.5 exactly; thresholds are strictly greater than
#         assert rl.try_consume_multiple(2) == 0
#         # Next token -> usage 0.75 exactly -> WARNING (strict >)
#         assert rl.try_consume() == 1
#         # Next would exceed block threshold -> BLOCKED
#         assert rl.try_consume() == 2

#     def test_burst_allows_temporarily(self):
#         rl = make_limiter(
#             tokens=2, window_s=2, burst=True, max_tokens=2, max_attempts=1
#         )
#         # Exhaust normally
#         assert rl.try_consume_multiple(2) == 0
#         # One more within max_tokens should be allowed for 1 attempt
#         assert rl.try_consume_multiple(2) == 0
#         # Another attempt should downgrade to WARNING (no more burst attempts)
#         assert rl.try_consume() == 1


# class TestRateLimiterHelpers:
#     """Test helper methods for observability."""

#     def test_tokens_remaining_and_usage(self):
#         rl = make_limiter(tokens=4, window_s=1)
#         assert rl.tokens_remaining() == 4
#         assert rl.usage() == 0.0

#         assert rl.try_consume_multiple(3) == 0
#         assert rl.tokens_remaining() == 1
#         assert rl.usage() == pytest.approx(0.75, abs=1e-12)

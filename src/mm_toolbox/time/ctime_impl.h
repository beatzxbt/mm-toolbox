/**
 * @file ctime_impl.h
 * @brief High-precision C time function declarations.
 *
 * Provides wall-clock and monotonic time functions using clock_gettime(),
 * as well as ISO 8601 timestamp formatting with automatic precision detection.
 * All functions are designed for high-frequency trading use cases with
 * minimal overhead.
 */

#ifndef CTIME_IMPL_H
#define CTIME_IMPL_H

#include <stdint.h>
#include <stddef.h>

/*
 * Wall-clock time functions using clock_gettime(CLOCK_REALTIME).
 * These may be affected by system clock adjustments (NTP, manual changes).
 */

/**
 * @brief Get current wall-clock time in seconds.
 * @return Time in seconds since Unix epoch, or -1 on error.
 */
int64_t c_time_s(void);

/**
 * @brief Get current wall-clock time in milliseconds.
 * @return Time in milliseconds since Unix epoch, or -1 on error.
 */
int64_t c_time_ms(void);

/**
 * @brief Get current wall-clock time in microseconds.
 * @return Time in microseconds since Unix epoch, or -1 on error.
 */
int64_t c_time_us(void);

/**
 * @brief Get current wall-clock time in nanoseconds.
 * @return Time in nanoseconds since Unix epoch, or -1 on error.
 */
int64_t c_time_ns(void);

/*
 * Monotonic time functions using clock_gettime(CLOCK_MONOTONIC).
 * Monotonic time never decreases and is unaffected by system clock changes.
 * Ideal for measuring elapsed time, timeouts, and performance timing.
 */

/**
 * @brief Get monotonic time in seconds.
 * @return Time in seconds, or -1 on error.
 */
int64_t c_time_monotonic_s(void);

/**
 * @brief Get monotonic time in milliseconds.
 * @return Time in milliseconds, or -1 on error.
 */
int64_t c_time_monotonic_ms(void);

/**
 * @brief Get monotonic time in microseconds.
 * @return Time in microseconds, or -1 on error.
 */
int64_t c_time_monotonic_us(void);

/**
 * @brief Get monotonic time in nanoseconds.
 * @return Time in nanoseconds, or -1 on error.
 */
int64_t c_time_monotonic_ns(void);

/* ISO 8601 conversion functions */

/**
 * @brief Format a Unix timestamp into an ISO 8601 string.
 *
 * Uses magnitude heuristics to detect input precision:
 *   - >= 1e18 : treated as nanoseconds
 *   - >= 1e15 : treated as microseconds
 *   - >= 1e12 : treated as milliseconds
 *   - else    : treated as seconds (float)
 *
 * @param timestamp  Unix timestamp (see precision rules above).
 * @param buf        Output buffer (must be at least 48 bytes for nanosecond precision).
 * @param buf_size   Size of output buffer.
 * @return           0 on success, -1 on error.
 * @note timestamp == 0.0 means "current time" for backward compatibility.
 *       To format the Unix epoch, use a small non-zero value like 1e-9.
 */
int c_time_iso8601(double timestamp, char* buf, size_t buf_size);

#endif /* CTIME_IMPL_H */

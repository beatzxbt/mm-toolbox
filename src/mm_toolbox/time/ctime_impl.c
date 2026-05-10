/**
 * @file ctime_impl.c
 * @brief High-precision C time implementation.
 *
 * Provides wall-clock and monotonic time functions using clock_gettime(),
 * as well as ISO 8601 timestamp formatting with automatic precision detection.
 * All functions are designed for high-frequency trading use cases with
 * minimal overhead.
 */

#include "ctime_impl.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>

/* Clock constants for high precision timing */
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

/**
 * @brief Get current wall-clock time in seconds.
 * @return Time in seconds since Unix epoch, or -1 on error.
 * @note Uses CLOCK_REALTIME which may be affected by system clock adjustments.
 */
int64_t c_time_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec;
}

/**
 * @brief Get current wall-clock time in milliseconds.
 * @return Time in milliseconds since Unix epoch, or -1 on error.
 * @note Uses CLOCK_REALTIME which may be affected by system clock adjustments.
 */
int64_t c_time_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000LL + (int64_t)ts.tv_nsec / 1000000LL;
}

/**
 * @brief Get current wall-clock time in microseconds.
 * @return Time in microseconds since Unix epoch, or -1 on error.
 * @note Uses CLOCK_REALTIME which may be affected by system clock adjustments.
 */
int64_t c_time_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000LL;
}

/**
 * @brief Get current wall-clock time in nanoseconds.
 * @return Time in nanoseconds since Unix epoch, or -1 on error.
 * @note Uses CLOCK_REALTIME which may be affected by system clock adjustments.
 */
int64_t c_time_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/**
 * @brief Get monotonic time in seconds.
 *
 * Monotonic time never decreases and is unaffected by system clock changes.
 * Ideal for measuring elapsed time and timeouts.
 *
 * @return Time in seconds, or -1 on error.
 */
int64_t c_time_monotonic_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec;
}

/**
 * @brief Get monotonic time in milliseconds.
 *
 * Monotonic time never decreases and is unaffected by system clock changes.
 * Ideal for measuring elapsed time and timeouts.
 *
 * @return Time in milliseconds, or -1 on error.
 */
int64_t c_time_monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000LL + (int64_t)ts.tv_nsec / 1000000LL;
}

/**
 * @brief Get monotonic time in microseconds.
 *
 * Monotonic time never decreases and is unaffected by system clock changes.
 * Ideal for measuring elapsed time and timeouts.
 *
 * @return Time in microseconds, or -1 on error.
 */
int64_t c_time_monotonic_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000LL;
}

/**
 * @brief Get monotonic time in nanoseconds.
 *
 * Monotonic time never decreases and is unaffected by system clock changes.
 * Ideal for measuring elapsed time, timeouts, and high-resolution performance timing.
 *
 * @return Time in nanoseconds, or -1 on error.
 */
int64_t c_time_monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

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
int c_time_iso8601(double timestamp, char* buf, size_t buf_size) {
    if (buf == NULL || buf_size == 0) {
        return -1;
    }
    
    if (timestamp == 0.0) {
        /* Fast path: get current time and format directly using manual arithmetic */
        const int64_t NS_IN_DAY = 86400000000000LL;
        const int64_t NS_IN_HOUR = 3600000000000LL;
        const int64_t NS_IN_MIN = 60000000000LL;
        const int64_t NS_IN_SEC = 1000000000LL;
        
        int64_t nanoseconds = c_time_ns();
        if (nanoseconds == -1) {
            return -1;
        }
        
        /* Convert nanoseconds to days + remainder */
        int64_t days_since_epoch = nanoseconds / NS_IN_DAY;
        int64_t remainder_ns = nanoseconds % NS_IN_DAY;
        
        /* Manual date arithmetic (Fliegel-Van Flandern algorithm) */
        int64_t z = days_since_epoch + 719468;
        int64_t era = (z >= 0) ? z / 146097 : (z - 146096) / 146097;
        int64_t doe = z - era * 146097;
        int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        int64_t year = yoe + era * 400;
        int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        int64_t mp = (5 * doy + 2) / 153;
        int64_t day = doy - (153 * mp + 2) / 5 + 1;
        int64_t month = (mp < 10) ? mp + 3 : mp - 9;
        
        if (month <= 2) {
            year += 1;
        }
        
        /* Convert remainder to time components */
        int64_t h = remainder_ns / NS_IN_HOUR;
        remainder_ns %= NS_IN_HOUR;
        int64_t M = remainder_ns / NS_IN_MIN;
        remainder_ns %= NS_IN_MIN;
        int64_t s = remainder_ns / NS_IN_SEC;
        remainder_ns %= NS_IN_SEC;
        int64_t ms = remainder_ns / 1000000LL;
        
        /* Format the result - 24 chars + null for millisecond precision */
        int ret = snprintf(buf, buf_size, "%04lld-%02lld-%02lldT%02lld:%02lld:%02lld.%03lldZ",
                 (long long)year, (long long)month, (long long)day,
                 (long long)h, (long long)M, (long long)s, (long long)ms);
        
        if (ret < 0 || (size_t)ret >= buf_size) {
            return -1;
        }
        
    } else {
        /* Provided timestamp path: use gmtime_r for thread safety */
        time_t seconds;
        int fractional_part;
        char fractional_str[16];
        struct tm utc_tm;
        struct tm* result;
        
        if (timestamp >= 1e18) {
            seconds = (time_t)(timestamp / 1e9);
            fractional_part = (int)((int64_t)timestamp % 1000000000);
            snprintf(fractional_str, sizeof(fractional_str), "%09d", fractional_part);
        } else if (timestamp >= 1e15) {
            seconds = (time_t)(timestamp / 1e6);
            fractional_part = (int)((int64_t)timestamp % 1000000);
            snprintf(fractional_str, sizeof(fractional_str), "%06d", fractional_part);
        } else if (timestamp >= 1e12) {
            seconds = (time_t)(timestamp / 1e3);
            fractional_part = (int)((int64_t)timestamp % 1000);
            snprintf(fractional_str, sizeof(fractional_str), "%03d", fractional_part);
        } else {
            seconds = (time_t)timestamp;
            fractional_part = (int)((timestamp - (int64_t)timestamp) * 1000);
            snprintf(fractional_str, sizeof(fractional_str), "%03d", fractional_part);
        }
        
        result = gmtime_r(&seconds, &utc_tm);
        if (result == NULL) {
            return -1;
        }
        
        int ret = snprintf(buf, buf_size, "%04d-%02d-%02dT%02d:%02d:%02d.%sZ",
                 utc_tm.tm_year + 1900, utc_tm.tm_mon + 1, utc_tm.tm_mday,
                 utc_tm.tm_hour, utc_tm.tm_min, utc_tm.tm_sec, fractional_str);
        
        if (ret < 0 || (size_t)ret >= buf_size) {
            return -1;
        }
    }
    
    return 0;
}

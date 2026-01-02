#include "ctime_impl.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

/* Clock constants for high precision timing */
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

int64_t c_time_s(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1; /* Error indicator */
    }
    return (int64_t)ts.tv_sec;
}

int64_t c_time_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1; /* Error indicator */
    }
    return (int64_t)ts.tv_sec * 1000LL + (int64_t)ts.tv_nsec / 1000000LL;
}

int64_t c_time_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1; /* Error indicator */
    }
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000LL;
}

int64_t c_time_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
        return -1; /* Error indicator */
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/**
 * Get monotonic time in seconds.
 * Monotonic time never decreases and is unaffected by system clock changes.
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
 * Get monotonic time in milliseconds.
 * Monotonic time never decreases and is unaffected by system clock changes.
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
 * Get monotonic time in microseconds.
 * Monotonic time never decreases and is unaffected by system clock changes.
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
 * Get monotonic time in nanoseconds.
 * Monotonic time never decreases and is unaffected by system clock changes.
 * @return Time in nanoseconds, or -1 on error.
 */
int64_t c_time_monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == -1) {
        return -1;
    }
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

char* c_time_iso8601(double timestamp) {
    char* buf = malloc(32);
    if (buf == NULL) {
        return NULL;
    }
    
    if (timestamp == 0.0) {
        /* Constants for time arithmetic */
        const int64_t NS_IN_DAY = 86400000000000LL;
        const int64_t NS_IN_HOUR = 3600000000000LL;
        const int64_t NS_IN_MIN = 60000000000LL;
        const int64_t NS_IN_SEC = 1000000000LL;
        
        /* Fast path: get current time and format directly using manual arithmetic */
        int64_t nanoseconds = c_time_ns();
        if (nanoseconds == -1) {
            free(buf);
            return NULL;
        }
        
        /* Convert nanoseconds to days + remainder */
        int64_t days_since_epoch = nanoseconds / NS_IN_DAY;
        int64_t remainder_ns = nanoseconds % NS_IN_DAY;
        
        /* Manual date arithmetic (Fliegel–Van Flandern algorithm) */
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
        
        /* Format the result */
        snprintf(buf, 32, "%04lld-%02lld-%02lldT%02lld:%02lld:%02lld.%03lldZ",
                 (long long)year, (long long)month, (long long)day,
                 (long long)h, (long long)M, (long long)s, (long long)ms);
        
    } else {
        /* Provided timestamp path: use gmtime for conversion */
        time_t seconds;
        int fractional_part;
        char fractional_str[10];
        
        if (timestamp >= 1e18) {  /* nanoseconds */
            seconds = (time_t)(timestamp / 1e9);
            fractional_part = (int)((int64_t)timestamp % 1000000000);
            snprintf(fractional_str, sizeof(fractional_str), "%09d", fractional_part);
        } else if (timestamp >= 1e15) {  /* microseconds */
            seconds = (time_t)(timestamp / 1e6);
            fractional_part = (int)((int64_t)timestamp % 1000000);
            snprintf(fractional_str, sizeof(fractional_str), "%06d", fractional_part);
        } else if (timestamp >= 1e12) {  /* milliseconds */
            seconds = (time_t)(timestamp / 1e3);
            fractional_part = (int)((int64_t)timestamp % 1000);
            snprintf(fractional_str, sizeof(fractional_str), "%03d", fractional_part);
        } else {  /* seconds */
            seconds = (time_t)timestamp;
            fractional_part = (int)((timestamp - (int64_t)timestamp) * 1000);
            snprintf(fractional_str, sizeof(fractional_str), "%03d", fractional_part);
        }
        
        struct tm* utc_tm = gmtime(&seconds);
        if (utc_tm == NULL) {
            free(buf);
            return NULL;
        }
        
        snprintf(buf, 32, "%04d-%02d-%02dT%02d:%02d:%02d.%sZ",
                 utc_tm->tm_year + 1900, utc_tm->tm_mon + 1, utc_tm->tm_mday,
                 utc_tm->tm_hour, utc_tm->tm_min, utc_tm->tm_sec, fractional_str);
    }
    
    return buf;
}

void c_free_string(char* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
} 

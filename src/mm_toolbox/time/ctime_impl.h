#ifndef CTIME_IMPL_H
#define CTIME_IMPL_H

#include <stdint.h>

/* High-performance wall-clock time functions using clock_gettime(CLOCK_REALTIME) */
int64_t c_time_s(void);
int64_t c_time_ms(void);
int64_t c_time_us(void);
int64_t c_time_ns(void);

/* High-performance monotonic time functions using clock_gettime(CLOCK_MONOTONIC).
 * Monotonic time never decreases and is unaffected by system clock changes.
 * Ideal for measuring elapsed time, timeouts, and performance timing. */
int64_t c_time_monotonic_s(void);
int64_t c_time_monotonic_ms(void);
int64_t c_time_monotonic_us(void);
int64_t c_time_monotonic_ns(void);

/* ISO8601 conversion functions */
char* c_time_iso8601(double timestamp);

/* Memory management */
void c_free_string(char* ptr);

#endif /* CTIME_IMPL_H */ 
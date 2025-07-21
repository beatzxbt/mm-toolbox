#ifndef CTIME_IMPL_H
#define CTIME_IMPL_H

#include <stdint.h>

/* High-performance time functions using clock_gettime directly */
int64_t c_time_s(void);
int64_t c_time_ms(void);
int64_t c_time_us(void);
int64_t c_time_ns(void);

/* ISO8601 conversion functions */
char* c_time_iso8601(double timestamp);

/* Memory management */
void c_free_string(char* ptr);

#endif /* CTIME_IMPL_H */ 
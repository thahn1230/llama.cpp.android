#pragma once

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Enable profiling by default for mobile builds
#ifndef GGML_PROFILING_ENABLED
#define GGML_PROFILING_ENABLED 1
#endif

#if GGML_PROFILING_ENABLED

// High-resolution timer functions
static inline double ggml_prof_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// Profiling data structure  
typedef struct {
    char name[64];
    double total_time_us;
    uint64_t call_count;
    uint64_t total_bytes;
    double min_time_us;
    double max_time_us;
    uint32_t layer_id;
    char layer_type[32];
} ggml_prof_stat_t;

typedef struct {
    ggml_prof_stat_t stats[64];
    int count;
    double session_start_time_us;
} ggml_profiler_t;

// Global profiler instance
extern ggml_profiler_t g_ggml_profiler;

// Profiling functions
void ggml_profiler_init(void);
void ggml_profiler_reset(void);
void ggml_profiler_print_results(void);
void ggml_profiler_save_results(const char* filename);
ggml_prof_stat_t* ggml_profiler_get_stat(const char* name);

// Timing macros
#define GGML_PROF_START(name, bytes) \
    double prof_start_##name = ggml_prof_time_us(); \
    ggml_prof_stat_t* prof_stat_##name = ggml_profiler_get_stat(#name); \
    uint64_t prof_bytes_##name = (uint64_t)(bytes);

#define GGML_PROF_END(name) \
    do { \
        double prof_end_time = ggml_prof_time_us(); \
        double prof_duration = prof_end_time - prof_start_##name; \
        if (prof_stat_##name) { \
            prof_stat_##name->total_time_us += prof_duration; \
            prof_stat_##name->call_count++; \
            prof_stat_##name->total_bytes += prof_bytes_##name; \
            if (prof_stat_##name->call_count == 1 || prof_duration < prof_stat_##name->min_time_us) { \
                prof_stat_##name->min_time_us = prof_duration; \
            } \
            if (prof_stat_##name->call_count == 1 || prof_duration > prof_stat_##name->max_time_us) { \
                prof_stat_##name->max_time_us = prof_duration; \
            } \
        } \
    } while(0)

// Specialized macros for different operation types
#define GGML_PROF_QUANTIZE_START(type, elements) \
    GGML_PROF_START(quantize_##type, (elements) * sizeof(float))

#define GGML_PROF_QUANTIZE_END(type) \
    GGML_PROF_END(quantize_##type)

#define GGML_PROF_VEC_DOT_START(type1, type2, n) \
    GGML_PROF_START(vec_dot_##type1##_##type2, (n) * (ggml_type_size(GGML_TYPE_##type1) + ggml_type_size(GGML_TYPE_##type2)))

#define GGML_PROF_VEC_DOT_END(type1, type2) \
    GGML_PROF_END(vec_dot_##type1##_##type2)

#define GGML_PROF_MATMUL_START(rows, cols) \
    GGML_PROF_START(matmul, (rows) * (cols) * sizeof(float))

#define GGML_PROF_MATMUL_END() \
    GGML_PROF_END(matmul)

#define GGML_PROF_MEMCPY_START(bytes) \
    GGML_PROF_START(memcpy, (bytes))

#define GGML_PROF_MEMCPY_END() \
    GGML_PROF_END(memcpy)

#define GGML_PROF_DEQUANT_START(type, elements) \
    GGML_PROF_START(dequant_##type, (elements) * ggml_type_size(GGML_TYPE_##type))

#define GGML_PROF_DEQUANT_END(type) \
    GGML_PROF_END(dequant_##type)

#else // GGML_PROFILING_ENABLED

// No-op macros when profiling is disabled
#define GGML_PROF_START(name, bytes)
#define GGML_PROF_END(name)
#define GGML_PROF_QUANTIZE_START(type, elements)
#define GGML_PROF_QUANTIZE_END(type)
#define GGML_PROF_VEC_DOT_START(type1, type2, n)
#define GGML_PROF_VEC_DOT_END(type1, type2)
#define GGML_PROF_MATMUL_START(rows, cols)
#define GGML_PROF_MATMUL_END()
#define GGML_PROF_MEMCPY_START(bytes)
#define GGML_PROF_MEMCPY_END()
#define GGML_PROF_DEQUANT_START(type, elements)
#define GGML_PROF_DEQUANT_END(type)

static inline void ggml_profiler_init(void) {}
static inline void ggml_profiler_reset(void) {}
static inline void ggml_profiler_print_results(void) {}
static inline void ggml_profiler_save_results(const char* filename) { (void)filename; }

#endif // GGML_PROFILING_ENABLED

#ifdef __cplusplus
}
#endif
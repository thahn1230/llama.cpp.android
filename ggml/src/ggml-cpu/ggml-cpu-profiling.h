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

// Block-scoped profiling macros to support nested profiling
#define GGML_PROF_START(name, bytes) \
    { \
        double ggml_prof_start_time = ggml_prof_time_us(); \
        ggml_prof_stat_t* ggml_prof_stat = ggml_profiler_get_stat(#name); \
        uint64_t ggml_prof_bytes = (uint64_t)(bytes);

#define GGML_PROF_END(name) \
        if (ggml_prof_stat) { \
            double ggml_prof_end_time = ggml_prof_time_us(); \
            double ggml_prof_duration = ggml_prof_end_time - ggml_prof_start_time; \
            ggml_prof_stat->total_time_us += ggml_prof_duration; \
            ggml_prof_stat->call_count++; \
            ggml_prof_stat->total_bytes += ggml_prof_bytes; \
            if (ggml_prof_stat->call_count == 1 || ggml_prof_duration < ggml_prof_stat->min_time_us) { \
                ggml_prof_stat->min_time_us = ggml_prof_duration; \
            } \
            if (ggml_prof_stat->call_count == 1 || ggml_prof_duration > ggml_prof_stat->max_time_us) { \
                ggml_prof_stat->max_time_us = ggml_prof_duration; \
            } \
        } \
    }

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

// Detailed dequantization profiling for w4a8 vs w8a8 analysis
#define GGML_PROF_W4_DEQUANT_START(bytes) GGML_PROF_START(w4_dequant, bytes)
#define GGML_PROF_W4_DEQUANT_END() GGML_PROF_END(w4_dequant)

#define GGML_PROF_MEMORY_LOAD_START(bytes) GGML_PROF_START(memory_load, bytes)
#define GGML_PROF_MEMORY_LOAD_END() GGML_PROF_END(memory_load)

#define GGML_PROF_DOT_COMPUTE_START(bytes) GGML_PROF_START(dot_compute, bytes)
#define GGML_PROF_DOT_COMPUTE_END() GGML_PROF_END(dot_compute)

// Transformer layer-specific profiling
#define GGML_PROF_RMSNORM_START(bytes) GGML_PROF_START(rmsnorm, bytes)
#define GGML_PROF_RMSNORM_END() GGML_PROF_END(rmsnorm)

#define GGML_PROF_ROPE_START(bytes) GGML_PROF_START(rope, bytes)
#define GGML_PROF_ROPE_END() GGML_PROF_END(rope)

#define GGML_PROF_SOFTMAX_START(bytes) GGML_PROF_START(softmax, bytes)
#define GGML_PROF_SOFTMAX_END() GGML_PROF_END(softmax)

#define GGML_PROF_ATTENTION_START(bytes) GGML_PROF_START(attention, bytes)
#define GGML_PROF_ATTENTION_END() GGML_PROF_END(attention)

// Layer projection profiling (Q, K, V, O, Up, Gate, Down)
#define GGML_PROF_Q_PROJ_START(bytes) GGML_PROF_START(q_projection, bytes)
#define GGML_PROF_Q_PROJ_END() GGML_PROF_END(q_projection)

#define GGML_PROF_K_PROJ_START(bytes) GGML_PROF_START(k_projection, bytes)
#define GGML_PROF_K_PROJ_END() GGML_PROF_END(k_projection)

#define GGML_PROF_V_PROJ_START(bytes) GGML_PROF_START(v_projection, bytes)
#define GGML_PROF_V_PROJ_END() GGML_PROF_END(v_projection)

#define GGML_PROF_O_PROJ_START(bytes) GGML_PROF_START(o_projection, bytes)
#define GGML_PROF_O_PROJ_END() GGML_PROF_END(o_projection)

#define GGML_PROF_UP_PROJ_START(bytes) GGML_PROF_START(up_projection, bytes)
#define GGML_PROF_UP_PROJ_END() GGML_PROF_END(up_projection)

#define GGML_PROF_GATE_PROJ_START(bytes) GGML_PROF_START(gate_projection, bytes)
#define GGML_PROF_GATE_PROJ_END() GGML_PROF_END(gate_projection)

#define GGML_PROF_DOWN_PROJ_START(bytes) GGML_PROF_START(down_projection, bytes)
#define GGML_PROF_DOWN_PROJ_END() GGML_PROF_END(down_projection)

// Enhanced function-specific profiling with source info
#define GGML_PROF_FUNC_START(func_name, bytes) GGML_PROF_START(func_name, bytes)
#define GGML_PROF_FUNC_END(func_name) GGML_PROF_END(func_name)

// Specific quantization format profiling  
#define GGML_PROF_Q4_K_START(bytes) GGML_PROF_START(q4_K_q8_K_w4a16, bytes)
#define GGML_PROF_Q4_K_END() GGML_PROF_END(q4_K_q8_K_w4a16)

#define GGML_PROF_Q4_0_START(bytes) GGML_PROF_START(q4_0_q8_0_w4a8, bytes)
#define GGML_PROF_Q4_0_END() GGML_PROF_END(q4_0_q8_0_w4a8)

#define GGML_PROF_Q4_1_START(bytes) GGML_PROF_START(q4_1_q8_1_w4a8, bytes)
#define GGML_PROF_Q4_1_END() GGML_PROF_END(q4_1_q8_1_w4a8)

// Source function identification
#define GGML_PROF_MUL_MAT_START(bytes) GGML_PROF_START(ggml_compute_forward_mul_mat, bytes)
#define GGML_PROF_MUL_MAT_END() GGML_PROF_END(ggml_compute_forward_mul_mat)

#define GGML_PROF_MUL_MAT_ID_START(bytes) GGML_PROF_START(ggml_compute_forward_mul_mat_id, bytes)
#define GGML_PROF_MUL_MAT_ID_END() GGML_PROF_END(ggml_compute_forward_mul_mat_id)

// Detailed ARM NEON vec_dot profiling
#define GGML_PROF_VEC_DOT_Q4_K_START(bytes) GGML_PROF_START(ggml_vec_dot_q4_K_q8_K_ARM, bytes)
#define GGML_PROF_VEC_DOT_Q4_K_END() GGML_PROF_END(ggml_vec_dot_q4_K_q8_K_ARM)

#define GGML_PROF_VEC_DOT_Q4_0_START(bytes) GGML_PROF_START(ggml_vec_dot_q4_0_q8_0_ARM, bytes)
#define GGML_PROF_VEC_DOT_Q4_0_END() GGML_PROF_END(ggml_vec_dot_q4_0_q8_0_ARM)

#define GGML_PROF_VEC_DOT_Q4_1_START(bytes) GGML_PROF_START(ggml_vec_dot_q4_1_q8_1_ARM, bytes)
#define GGML_PROF_VEC_DOT_Q4_1_END() GGML_PROF_END(ggml_vec_dot_q4_1_q8_1_ARM)

#define GGML_PROF_VEC_DOT_Q8_0_START(bytes) GGML_PROF_START(ggml_vec_dot_q8_0_q8_0_ARM, bytes)
#define GGML_PROF_VEC_DOT_Q8_0_END() GGML_PROF_END(ggml_vec_dot_q8_0_q8_0_ARM)

// Layer-specific profiling with source function names
#define GGML_PROF_RMSNORM_FUNC_START(bytes) GGML_PROF_START(ggml_compute_forward_rms_norm_f32, bytes)
#define GGML_PROF_RMSNORM_FUNC_END() GGML_PROF_END(ggml_compute_forward_rms_norm_f32)

#define GGML_PROF_ROPE_FUNC_START(bytes) GGML_PROF_START(ggml_compute_forward_rope_f32, bytes)
#define GGML_PROF_ROPE_FUNC_END() GGML_PROF_END(ggml_compute_forward_rope_f32)

#define GGML_PROF_SOFTMAX_FUNC_START(bytes) GGML_PROF_START(ggml_compute_forward_soft_max_f32, bytes)
#define GGML_PROF_SOFTMAX_FUNC_END() GGML_PROF_END(ggml_compute_forward_soft_max_f32)

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
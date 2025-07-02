#include "ggml/src/ggml-cpu/ggml-cpu-profiling.h"
#include "ggml/include/ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Mock quantization functions for testing ARM NEON profiling
void test_quantize_q8_0(const float* x, void* y, int k) {
    GGML_PROF_FUNC_START(quantize_row_q8_0_ARM, k * sizeof(float));
    
    // Simulate work
    volatile float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += x[i] * 0.125f;
    }
    
    GGML_PROF_FUNC_END(quantize_row_q8_0_ARM);
}

void test_vec_dot_q4_0_q8_0(int n, float* s, const void* vx, const void* vy) {
    GGML_PROF_VEC_DOT_Q4_0_START(n * 6); // 4-bit weight + 8-bit activation
    
    // Simulate w4a8 dequantization + dot product work
    GGML_PROF_W4_DEQUANT_START(n * 4);
    
    volatile float dequant_sum = 0.0f;
    for (int i = 0; i < n/32; i++) {
        dequant_sum += (float)i * 0.0625f; // 4-bit → 8-bit simulation
    }
    
    GGML_PROF_W4_DEQUANT_END();
    
    GGML_PROF_DOT_COMPUTE_START(n * 2);
    
    volatile float dot_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dot_sum += (float)i * (float)(i+1);
    }
    *s = dot_sum;
    
    GGML_PROF_DOT_COMPUTE_END();
    
    GGML_PROF_VEC_DOT_Q4_0_END();
}

void test_vec_dot_q8_0_q8_0(int n, float* s, const void* vx, const void* vy) {
    GGML_PROF_VEC_DOT_Q8_0_START(n * 2); // 8-bit weight + 8-bit activation
    
    // Simulate w8a8 direct computation (no dequantization)
    GGML_PROF_DOT_COMPUTE_START(n * 2);
    
    volatile float dot_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dot_sum += (float)i * (float)(i+1) * 0.5f;
    }
    *s = dot_sum;
    
    GGML_PROF_DOT_COMPUTE_END();
    
    GGML_PROF_VEC_DOT_Q8_0_END();
}

void test_vec_dot_q4_K_q8_K(int n, float* s, const void* vx, const void* vy) {
    GGML_PROF_VEC_DOT_Q4_K_START(n * 6); // 4-bit weight + 16-bit activation
    
    // Simulate w4a16 work with more complex dequantization
    GGML_PROF_W4_DEQUANT_START(n * 4);
    
    volatile float dequant_sum = 0.0f;
    for (int i = 0; i < n/16; i++) {
        dequant_sum += (float)i * 0.0625f * 2.0f; // More complex 4-bit → 16-bit
    }
    
    GGML_PROF_W4_DEQUANT_END();
    
    GGML_PROF_DOT_COMPUTE_START(n * 3);
    
    volatile float dot_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        dot_sum += (float)i * (float)(i+1) * 1.5f;
    }
    *s = dot_sum;
    
    GGML_PROF_DOT_COMPUTE_END();
    
    GGML_PROF_VEC_DOT_Q4_K_END();
}

int main() {
    printf("=== ARM NEON Quantization Profiling Test ===\n");
    
    // Initialize profiler
    ggml_profiler_init();
    
    const int test_size = 4096;
    const int iterations = 1000;
    
    // Allocate test data
    float* input_data = (float*)malloc(test_size * sizeof(float));
    float* output_data = (float*)malloc(test_size * sizeof(float));
    void* quantized_data = malloc(test_size * 2);
    
    // Initialize test data
    for (int i = 0; i < test_size; i++) {
        input_data[i] = (float)i / test_size;
    }
    
    printf("\nRunning %d iterations with %d elements each...\n", iterations, test_size);
    
    // Test 1: Q8_0 Quantization (8-bit activation prep)
    printf("\n1. Testing Q8_0 quantization (activation prep)...\n");
    for (int i = 0; i < iterations/4; i++) {
        test_quantize_q8_0(input_data, quantized_data, test_size);
    }
    
    // Test 2: Q4_0 → Q8_0 Vector Dot (w4a8)
    printf("2. Testing Q4_0→Q8_0 vector dot (w4a8)...\n");
    for (int i = 0; i < iterations; i++) {
        float result;
        test_vec_dot_q4_0_q8_0(test_size, &result, quantized_data, quantized_data);
    }
    
    // Test 3: Q8_0 → Q8_0 Vector Dot (w8a8)
    printf("3. Testing Q8_0→Q8_0 vector dot (w8a8)...\n");
    for (int i = 0; i < iterations; i++) {
        float result;
        test_vec_dot_q8_0_q8_0(test_size, &result, quantized_data, quantized_data);
    }
    
    // Test 4: Q4_K → Q8_K Vector Dot (w4a16)
    printf("4. Testing Q4_K→Q8_K vector dot (w4a16)...\n");
    for (int i = 0; i < iterations; i++) {
        float result;
        test_vec_dot_q4_K_q8_K(test_size, &result, quantized_data, quantized_data);
    }
    
    printf("\n=== ARM NEON Profiling Results ===\n");
    ggml_profiler_print_results();
    
    // Cleanup
    free(input_data);
    free(output_data);
    free(quantized_data);
    
    return 0;
} 
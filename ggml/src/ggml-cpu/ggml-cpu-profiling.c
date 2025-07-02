#include "ggml-cpu-profiling.h"
#include "ggml.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if GGML_PROFILING_ENABLED

// Global profiler instance
ggml_profiler_t g_ggml_profiler = {0};

// Thread-local profiling stack
__thread ggml_prof_ctx_t ggml_prof_stack[GGML_MAX_PROF_DEPTH] = {0};
__thread int ggml_prof_stack_depth = 0;

void ggml_profiler_init(void) {
    if (g_ggml_profiler.initialized) {
        return;
    }
    
    memset(&g_ggml_profiler, 0, sizeof(ggml_profiler_t));
    pthread_mutex_init(&g_ggml_profiler.mutex, NULL);
    g_ggml_profiler.session_start_time_us = ggml_prof_time_us();
    g_ggml_profiler.initialized = 1;
    printf("[GGML PROFILER] Profiling initialized\n");
}

void ggml_profiler_reset(void) {
    if (!g_ggml_profiler.initialized) {
        return;
    }
    
    pthread_mutex_lock(&g_ggml_profiler.mutex);
    g_ggml_profiler.count = 0;
    memset(g_ggml_profiler.stats, 0, sizeof(g_ggml_profiler.stats));
    g_ggml_profiler.session_start_time_us = ggml_prof_time_us();
    pthread_mutex_unlock(&g_ggml_profiler.mutex);
    printf("[GGML PROFILER] Profiling reset\n");
}

ggml_prof_stat_t* ggml_profiler_get_stat(const char* name) {
    if (!name || !g_ggml_profiler.initialized) return NULL;
    
    // Try to find existing stat
    for (int i = 0; i < g_ggml_profiler.count; i++) {
        if (strcmp(g_ggml_profiler.stats[i].name, name) == 0) {
            return &g_ggml_profiler.stats[i];
        }
    }
    
    // Create new stat if we have space
    if (g_ggml_profiler.count < 128) {  // Increased from 64 to 128
        ggml_prof_stat_t* stat = &g_ggml_profiler.stats[g_ggml_profiler.count];
        strncpy(stat->name, name, sizeof(stat->name) - 1);
        stat->name[sizeof(stat->name) - 1] = '\0';
        stat->total_time_us = 0.0;
        stat->call_count = 0;
        stat->total_bytes = 0;
        stat->min_time_us = 0.0;
        stat->max_time_us = 0.0;
        g_ggml_profiler.count++;
        return stat;
    }
    
    return NULL;
}

static void print_separator(void) {
    printf("================================================================================\n");
}

static void print_header(void) {
    printf("%-20s %10s %12s %12s %12s %12s %8s\n", 
           "Operation", "Calls", "Total(ms)", "Avg(μs)", "Min(μs)", "Max(μs)", "MB/s");
    print_separator();
}

static double calculate_bandwidth_mbps(uint64_t bytes, double time_us) {
    if (time_us <= 0.0) return 0.0;
    return (bytes / 1024.0 / 1024.0) / (time_us / 1000000.0);
}

void ggml_profiler_print_results(void) {
    if (g_ggml_profiler.count == 0) {
        printf("[GGML PROFILER] No profiling data available\n");
        return;
    }
    
    double session_total_time_us = ggml_prof_time_us() - g_ggml_profiler.session_start_time_us;
    
    printf("\n");
    print_separator();
    printf("                           GGML PROFILING RESULTS\n");
    print_separator();
    printf("Session Duration: %.2f ms\n", session_total_time_us / 1000.0);
    printf("Total Operations: %d\n", g_ggml_profiler.count);
    print_separator();
    
    print_header();
    
    double total_all_ops_time_us = 0.0;
    uint64_t total_all_bytes = 0;
    
    // Sort operations by total time (descending)
    ggml_prof_stat_t sorted_stats[64];
    memcpy(sorted_stats, g_ggml_profiler.stats, g_ggml_profiler.count * sizeof(ggml_prof_stat_t));
    
    for (int i = 0; i < g_ggml_profiler.count - 1; i++) {
        for (int j = i + 1; j < g_ggml_profiler.count; j++) {
            if (sorted_stats[i].total_time_us < sorted_stats[j].total_time_us) {
                ggml_prof_stat_t temp = sorted_stats[i];
                sorted_stats[i] = sorted_stats[j];
                sorted_stats[j] = temp;
            }
        }
    }
    
    for (int i = 0; i < g_ggml_profiler.count; i++) {
        ggml_prof_stat_t* stat = &sorted_stats[i];
        if (stat->call_count == 0) continue;
        
        double avg_time_us = stat->total_time_us / stat->call_count;
        double bandwidth_mbps = calculate_bandwidth_mbps(stat->total_bytes, stat->total_time_us);
        
        printf("%-20s %10lu %12.2f %12.2f %12.2f %12.2f %8.1f\n",
               stat->name,
               stat->call_count,
               stat->total_time_us / 1000.0,  // Convert to ms
               avg_time_us,
               stat->min_time_us,
               stat->max_time_us,
               bandwidth_mbps);
        
        total_all_ops_time_us += stat->total_time_us;
        total_all_bytes += stat->total_bytes;
    }
    
    print_separator();
    printf("%-20s %10s %12.2f %12s %12s %12s %8.1f\n",
           "TOTAL", "-", 
           total_all_ops_time_us / 1000.0,
           "-", "-", "-",
           calculate_bandwidth_mbps(total_all_bytes, total_all_ops_time_us));
    
    printf("Profiling Overhead: %.2f%% of session time\n", 
           (total_all_ops_time_us / session_total_time_us) * 100.0);
    print_separator();
    
    // Print operation type breakdown
    printf("\nOperation Type Breakdown:\n");
    print_separator();
    
    struct type_summary {
        char name[32];
        double total_time_us;
        uint64_t call_count;
        uint64_t total_bytes;
    } type_summaries[10] = {0};
    int type_count = 0;
    
    const char* type_prefixes[] = {"quantize", "vec_dot", "matmul", "memcpy", "dequant"};
    const int num_prefixes = sizeof(type_prefixes) / sizeof(type_prefixes[0]);
    
    for (int p = 0; p < num_prefixes; p++) {
        struct type_summary* summary = &type_summaries[type_count];
        strncpy(summary->name, type_prefixes[p], sizeof(summary->name) - 1);
        
        for (int i = 0; i < g_ggml_profiler.count; i++) {
            if (strncmp(g_ggml_profiler.stats[i].name, type_prefixes[p], strlen(type_prefixes[p])) == 0) {
                summary->total_time_us += g_ggml_profiler.stats[i].total_time_us;
                summary->call_count += g_ggml_profiler.stats[i].call_count;
                summary->total_bytes += g_ggml_profiler.stats[i].total_bytes;
            }
        }
        
        if (summary->call_count > 0) {
            type_count++;
        }
    }
    
    for (int i = 0; i < type_count; i++) {
        struct type_summary* summary = &type_summaries[i];
        double percentage = (summary->total_time_us / total_all_ops_time_us) * 100.0;
        printf("%-15s: %8.2f ms (%5.1f%%) - %lu calls - %.1f MB/s\n",
               summary->name,
               summary->total_time_us / 1000.0,
               percentage,
               summary->call_count,
               calculate_bandwidth_mbps(summary->total_bytes, summary->total_time_us));
    }
    
    print_separator();
    
         // w4a8 vs w8a8 Detailed Analysis  
     printf("\n🔬 w4a8 vs w8a8 Dequantization Analysis:\n");
     print_separator();
     
     // Debug: Show all quantization-related operations found
     printf("DEBUG - Found quantization operations:\n");
     for (int i = 0; i < g_ggml_profiler.count; i++) {
         const char* name = g_ggml_profiler.stats[i].name;
         if (strstr(name, "q4") || strstr(name, "q8") || strstr(name, "vec_dot") || 
             strstr(name, "dequant") || strstr(name, "memory_load") || strstr(name, "dot_compute")) {
             printf("  - %s: %lu calls\n", name, g_ggml_profiler.stats[i].call_count);
         }
     }
     printf("\n");
    
    ggml_prof_stat_t* w4_dequant = NULL;
    ggml_prof_stat_t* memory_load = NULL;
    ggml_prof_stat_t* dot_compute = NULL;
    ggml_prof_stat_t* q4_w4a8 = NULL;
    ggml_prof_stat_t* q8_w8a8 = NULL;
    
         // Find relevant stats
     ggml_prof_stat_t* q4_K_w4a16 = NULL;
     ggml_prof_stat_t* q4_0_w4a8 = NULL;
     ggml_prof_stat_t* q4_1_w4a8 = NULL;
     
     for (int i = 0; i < g_ggml_profiler.count; i++) {
         if (strcmp(g_ggml_profiler.stats[i].name, "w4_dequant") == 0) w4_dequant = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "memory_load") == 0) memory_load = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "dot_compute") == 0) dot_compute = &g_ggml_profiler.stats[i];
         // New function-specific names
         else if (strcmp(g_ggml_profiler.stats[i].name, "ggml_vec_dot_q4_1_q8_1_ARM") == 0) q4_1_w4a8 = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "ggml_vec_dot_q4_0_q8_0_ARM") == 0) q4_0_w4a8 = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "ggml_vec_dot_q4_K_q8_K_ARM") == 0) q4_K_w4a16 = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "ggml_vec_dot_q8_0_q8_0_ARM") == 0) q8_w8a8 = &g_ggml_profiler.stats[i];
         // Legacy names for backward compatibility
         else if (strcmp(g_ggml_profiler.stats[i].name, "q4_1_q8_1_w4a8") == 0) q4_w4a8 = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "q8_0_q8_0_w8a8") == 0 && !q8_w8a8) q8_w8a8 = &g_ggml_profiler.stats[i];
     }
    
    if (w4_dequant && w4_dequant->call_count > 0) {
        printf("W4 Dequantization  : %8.2f ms (%6lu calls) - %.1f MB/s\n", 
               w4_dequant->total_time_us / 1000.0, w4_dequant->call_count, 
               calculate_bandwidth_mbps(w4_dequant->total_bytes, w4_dequant->total_time_us));
    }
    if (memory_load && memory_load->call_count > 0) {
        printf("Memory Load        : %8.2f ms (%6lu calls) - %.1f MB/s\n", 
               memory_load->total_time_us / 1000.0, memory_load->call_count,
               calculate_bandwidth_mbps(memory_load->total_bytes, memory_load->total_time_us));
    }
    if (dot_compute && dot_compute->call_count > 0) {
        printf("Dot Computation    : %8.2f ms (%6lu calls) - %.1f MB/s\n", 
               dot_compute->total_time_us / 1000.0, dot_compute->call_count,
               calculate_bandwidth_mbps(dot_compute->total_bytes, dot_compute->total_time_us));
    }
         // Display quantization format details with function mapping
     if (q4_K_w4a16 && q4_K_w4a16->call_count > 0) {
         printf("Q4_K (w4a16)       : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_vec_dot_q4_K_q8_K]\n", 
                q4_K_w4a16->total_time_us / 1000.0, q4_K_w4a16->call_count,
                calculate_bandwidth_mbps(q4_K_w4a16->total_bytes, q4_K_w4a16->total_time_us));
     }
     if (q4_1_w4a8 && q4_1_w4a8->call_count > 0) {
         printf("Q4_1 (w4a8)        : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_vec_dot_q4_1_q8_1]\n", 
                q4_1_w4a8->total_time_us / 1000.0, q4_1_w4a8->call_count,
                calculate_bandwidth_mbps(q4_1_w4a8->total_bytes, q4_1_w4a8->total_time_us));
     }
     if (q4_0_w4a8 && q4_0_w4a8->call_count > 0) {
         printf("Q4_0 (w4a8)        : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_vec_dot_q4_0_q8_0]\n", 
                q4_0_w4a8->total_time_us / 1000.0, q4_0_w4a8->call_count,
                calculate_bandwidth_mbps(q4_0_w4a8->total_bytes, q4_0_w4a8->total_time_us));
     }
     if (q8_w8a8 && q8_w8a8->call_count > 0) {
         printf("Q8_0 (w8a8)        : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_vec_dot_q8_0_q8_0]\n", 
                q8_w8a8->total_time_us / 1000.0, q8_w8a8->call_count,
                calculate_bandwidth_mbps(q8_w8a8->total_bytes, q8_w8a8->total_time_us));
     }
     
     // Performance comparison - prioritize most used formats
     ggml_prof_stat_t* w4_format = q4_K_w4a16 ? q4_K_w4a16 : (q4_1_w4a8 ? q4_1_w4a8 : q4_0_w4a8);
     if (w4_format && q8_w8a8 && w4_format->call_count > 0 && q8_w8a8->call_count > 0) {
         double w4_avg = w4_format->total_time_us / w4_format->call_count;
         double w8_avg = q8_w8a8->total_time_us / q8_w8a8->call_count;
         printf("Performance Ratio  : w8a8 is %.2fx %s than w4 formats\n", 
                fabs(w4_avg / w8_avg), 
                (w8_avg < w4_avg) ? "FASTER" : "SLOWER");
     }
    
         // Transformer Layer Analysis
     printf("\n🧠 Transformer Layer Analysis:\n");
     print_separator();
     
     ggml_prof_stat_t* rmsnorm = NULL;
     ggml_prof_stat_t* rope = NULL;
     ggml_prof_stat_t* softmax = NULL;
     ggml_prof_stat_t* q_proj = NULL;
     ggml_prof_stat_t* k_proj = NULL;
     ggml_prof_stat_t* v_proj = NULL;
     ggml_prof_stat_t* o_proj = NULL;
     ggml_prof_stat_t* up_proj = NULL;
     ggml_prof_stat_t* gate_proj = NULL;
     ggml_prof_stat_t* down_proj = NULL;
    
         for (int i = 0; i < g_ggml_profiler.count; i++) {
         // Layer operations
         if (strcmp(g_ggml_profiler.stats[i].name, "ggml_compute_forward_rms_norm_f32") == 0) rmsnorm = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "ggml_compute_forward_rope_f32") == 0) rope = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "ggml_compute_forward_soft_max_f32") == 0) softmax = &g_ggml_profiler.stats[i];
         // Projection operations
         else if (strcmp(g_ggml_profiler.stats[i].name, "q_projection") == 0) q_proj = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "k_projection") == 0) k_proj = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "v_projection") == 0) v_proj = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "o_projection") == 0) o_proj = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "up_projection") == 0) up_proj = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "gate_projection") == 0) gate_proj = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "down_projection") == 0) down_proj = &g_ggml_profiler.stats[i];
         // Legacy names for backward compatibility
         else if (strcmp(g_ggml_profiler.stats[i].name, "rmsnorm") == 0 && !rmsnorm) rmsnorm = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "rope") == 0 && !rope) rope = &g_ggml_profiler.stats[i];
         else if (strcmp(g_ggml_profiler.stats[i].name, "softmax") == 0 && !softmax) softmax = &g_ggml_profiler.stats[i];
     }
     
     if (rmsnorm && rmsnorm->call_count > 0) {
         printf("RMSNorm            : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_compute_forward_rms_norm]\n", 
                rmsnorm->total_time_us / 1000.0, rmsnorm->call_count,
                calculate_bandwidth_mbps(rmsnorm->total_bytes, rmsnorm->total_time_us));
     }
     if (rope && rope->call_count > 0) {
         printf("RoPE               : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_compute_forward_rope]\n", 
                rope->total_time_us / 1000.0, rope->call_count,
                calculate_bandwidth_mbps(rope->total_bytes, rope->total_time_us));
     }
     if (softmax && softmax->call_count > 0) {
         printf("Softmax            : %8.2f ms (%6lu calls) - %.1f MB/s [ggml_compute_forward_soft_max]\n", 
                softmax->total_time_us / 1000.0, softmax->call_count,
                calculate_bandwidth_mbps(softmax->total_bytes, softmax->total_time_us));
     }
     
     // Attention projections
     if (q_proj && q_proj->call_count > 0) {
         printf("Q Projection       : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:q]\n", 
                q_proj->total_time_us / 1000.0, q_proj->call_count,
                calculate_bandwidth_mbps(q_proj->total_bytes, q_proj->total_time_us));
     }
     if (k_proj && k_proj->call_count > 0) {
         printf("K Projection       : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:k]\n", 
                k_proj->total_time_us / 1000.0, k_proj->call_count,
                calculate_bandwidth_mbps(k_proj->total_bytes, k_proj->total_time_us));
     }
     if (v_proj && v_proj->call_count > 0) {
         printf("V Projection       : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:v]\n", 
                v_proj->total_time_us / 1000.0, v_proj->call_count,
                calculate_bandwidth_mbps(v_proj->total_bytes, v_proj->total_time_us));
     }
     if (o_proj && o_proj->call_count > 0) {
         printf("O Projection       : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:o]\n", 
                o_proj->total_time_us / 1000.0, o_proj->call_count,
                calculate_bandwidth_mbps(o_proj->total_bytes, o_proj->total_time_us));
     }
     
     // FFN projections
     if (up_proj && up_proj->call_count > 0) {
         printf("Up Projection      : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:up]\n", 
                up_proj->total_time_us / 1000.0, up_proj->call_count,
                calculate_bandwidth_mbps(up_proj->total_bytes, up_proj->total_time_us));
     }
     if (gate_proj && gate_proj->call_count > 0) {
         printf("Gate Projection    : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:gate]\n", 
                gate_proj->total_time_us / 1000.0, gate_proj->call_count,
                calculate_bandwidth_mbps(gate_proj->total_bytes, gate_proj->total_time_us));
     }
     if (down_proj && down_proj->call_count > 0) {
         printf("Down Projection    : %8.2f ms (%6lu calls) - %.1f MB/s [mul_mat:down]\n", 
                down_proj->total_time_us / 1000.0, down_proj->call_count,
                calculate_bandwidth_mbps(down_proj->total_bytes, down_proj->total_time_us));
     }
    
    print_separator();
    printf("\n");
}

void ggml_profiler_save_results(const char* filename) {
    if (!filename || g_ggml_profiler.count == 0) {
        return;
    }
    
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("[GGML PROFILER] Failed to open file %s for writing\n", filename);
        return;
    }
    
    double session_total_time_us = ggml_prof_time_us() - g_ggml_profiler.session_start_time_us;
    
    fprintf(file, "# GGML Profiling Results\n");
    fprintf(file, "# Session Duration: %.2f ms\n", session_total_time_us / 1000.0);
    fprintf(file, "# Total Operations: %d\n", g_ggml_profiler.count);
    fprintf(file, "Operation,Calls,Total_ms,Avg_us,Min_us,Max_us,Total_Bytes,Bandwidth_MBps\n");
    
    for (int i = 0; i < g_ggml_profiler.count; i++) {
        ggml_prof_stat_t* stat = &g_ggml_profiler.stats[i];
        if (stat->call_count == 0) continue;
        
        double avg_time_us = stat->total_time_us / stat->call_count;
        double bandwidth_mbps = calculate_bandwidth_mbps(stat->total_bytes, stat->total_time_us);
        
        fprintf(file, "%s,%lu,%.2f,%.2f,%.2f,%.2f,%lu,%.1f\n",
                stat->name,
                stat->call_count,
                stat->total_time_us / 1000.0,
                avg_time_us,
                stat->min_time_us,
                stat->max_time_us,
                stat->total_bytes,
                bandwidth_mbps);
    }
    
    fclose(file);
    printf("[GGML PROFILER] Results saved to %s\n", filename);
}

#endif // GGML_PROFILING_ENABLED

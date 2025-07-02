#include "ggml-cpu-profiling.h"
#include "ggml.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if GGML_PROFILING_ENABLED

// Global profiler instance
ggml_profiler_t g_ggml_profiler = {0};

void ggml_profiler_init(void) {
    memset(&g_ggml_profiler, 0, sizeof(ggml_profiler_t));
    g_ggml_profiler.session_start_time_us = ggml_prof_time_us();
    printf("[GGML PROFILER] Profiling initialized\n");
}

void ggml_profiler_reset(void) {
    g_ggml_profiler.count = 0;
    memset(g_ggml_profiler.stats, 0, sizeof(g_ggml_profiler.stats));
    g_ggml_profiler.session_start_time_us = ggml_prof_time_us();
    printf("[GGML PROFILER] Profiling reset\n");
}

ggml_prof_stat_t* ggml_profiler_get_stat(const char* name) {
    if (!name) return NULL;
    
    // Try to find existing stat
    for (int i = 0; i < g_ggml_profiler.count; i++) {
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
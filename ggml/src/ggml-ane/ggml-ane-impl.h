// ggml-ane-impl.h - Internal implementation structures for ANE backend
#pragma once

#include "ggml-ane.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <unordered_map>
#include <mutex>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#endif

// Forward declaration for kernel structure
struct ggml_ane_kernel;
typedef struct ggml_ane_kernel * ggml_ane_kernel_t;

////////////////////////////////////////////////////////////////////////////////
// ANE Runtime API (ggml-ane-runtime.mm)
////////////////////////////////////////////////////////////////////////////////

// Initialize ANE runtime - loads private framework
bool ggml_ane_runtime_init(void);

// Check if ANE is available
bool ggml_ane_is_available(void);

// Get current compile count
int ggml_ane_get_compile_count(void);

// Check if process restart is needed (due to ~119 compile limit)
bool ggml_ane_needs_restart(void);

// Compile MIL program with weights into executable kernel
// hash is optional - if provided, kernel will be cached
ggml_ane_kernel_t ggml_ane_compile_kernel(
    const char * mil_text,
    const void * weights,
    size_t weights_size,
    int n_inputs,
    size_t * input_sizes,
    int n_outputs,
    size_t * output_sizes,
    uint64_t hash
);

// Write data to kernel input
void ggml_ane_write_input(ggml_ane_kernel_t kernel, int index, const void * data, size_t bytes);

// Read data from kernel output
void ggml_ane_read_output(ggml_ane_kernel_t kernel, int index, void * data, size_t bytes);

// Execute kernel (assumes inputs already written)
bool ggml_ane_eval_kernel(ggml_ane_kernel_t kernel);

// Convenience: write inputs, execute, read outputs
bool ggml_ane_execute(
    ggml_ane_kernel_t kernel,
    const void ** inputs,
    void ** outputs
);

// Free kernel
void ggml_ane_free_kernel(ggml_ane_kernel_t kernel);

// Clear kernel cache
void ggml_ane_clear_cache(void);

// Get kernel I/O sizes
size_t ggml_ane_get_kernel_input_size(ggml_ane_kernel_t kernel, int index);
size_t ggml_ane_get_kernel_output_size(ggml_ane_kernel_t kernel, int index);
int ggml_ane_get_kernel_input_count(ggml_ane_kernel_t kernel);
int ggml_ane_get_kernel_output_count(ggml_ane_kernel_t kernel);

////////////////////////////////////////////////////////////////////////////////
// MIL Generation API (ggml-ane-mil.mm)
////////////////////////////////////////////////////////////////////////////////

#ifdef __OBJC__
// Build FP16 weight blob with header
NSData * ggml_ane_build_weight_blob(const float * weights_f32, int out_ch, int in_ch);

// Generate MIL for operations
NSString * ggml_ane_gen_mil_matmul(int in_ch, int out_ch, int spatial);
NSString * ggml_ane_gen_mil_conv(int in_ch, int out_ch, int spatial);
NSString * ggml_ane_gen_mil_add(int channels, int spatial);
NSString * ggml_ane_gen_mil_mul(int channels, int spatial);
NSString * ggml_ane_gen_mil_silu(int channels, int spatial);
NSString * ggml_ane_gen_mil_softmax(int channels, int spatial);
NSString * ggml_ane_gen_mil_rms_norm(int dim, int spatial, float eps);

// Convert NSString to C string (caller must free)
char * ggml_ane_mil_to_cstring(NSString * mil);
#endif

// ANE device context
struct ggml_ane_device_context {
    int device_index;
    char name[64];
    char description[256];
    size_t memory_size;          // Unified memory available
    int n_neural_cores;          // Should be 16 on M-series
    bool supports_fp16;          // Always true on ANE
    bool supports_int8;          // True but same speed as fp16
};

// ANE buffer type context
struct ggml_ane_buffer_type_context {
    ggml_backend_dev_t device;
    size_t alignment;            // IOSurface alignment requirements
    size_t max_size;             // Maximum buffer size
};

// ANE buffer context
struct ggml_ane_buffer_context {
    ggml_backend_buffer_type_t buft;
    void * base;                 // Base address (mapped IOSurface)
    size_t size;                 // Total size
    size_t allocated_size;       // Actually allocated
#ifdef __OBJC__
    IOSurfaceRef surface;        // Underlying IOSurface
#endif
};

// ANE backend context
struct ggml_ane_context {
    ggml_backend_dev_t device;
    
    // Kernel cache
    std::unordered_map<uint64_t, void*> kernel_cache;
    std::mutex kernel_cache_mutex;
    
    // Performance stats
    struct ggml_ane_perf_stats perf_stats;
    
    // Compile counter (workaround for ~119 compile limit)
    int compile_count;
    bool needs_restart;
};

// Graph plan context
struct ggml_ane_graph_plan {
    struct ggml_cgraph * graph;
    
    // Compiled kernels for each supported node
    std::vector<void*> kernels;
    
    // Node execution order (indices into graph->nodes)
    std::vector<int> node_order;
    
    // Which nodes are ANE-supported
    std::vector<bool> node_supported;
    
    // Input/output surfaces for execution
    std::vector<void*> io_surfaces;
    
    // Memory requirements
    size_t total_memory;
};

// MIL generation context
struct ggml_ane_mil_context {
    char * mil_text;
    size_t mil_text_size;
    size_t mil_text_capacity;
    
    void * weight_blob;
    size_t weight_blob_size;
    
    int n_inputs;
    int n_outputs;
    size_t input_sizes[16];
    size_t output_sizes[16];
};

// Hash function for tensor shapes (for kernel caching)
static inline uint64_t ggml_ane_hash_shape(const int64_t * ne, int n_dims) {
    uint64_t hash = 0xcbf29ce484222325ULL; // FNV offset basis
    for (int i = 0; i < n_dims; i++) {
        hash ^= (uint64_t)ne[i];
        hash *= 0x100000001b3ULL; // FNV prime
    }
    return hash;
}

// Logging macros
#define GGML_ANE_LOG_DEBUG(...) do { \
    if (getenv("GGML_ANE_DEBUG")) { \
        fprintf(stderr, "[ANE DEBUG] " __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while(0)

#define GGML_ANE_LOG_INFO(...) do { \
    fprintf(stderr, "[ANE] " __VA_ARGS__); \
    fprintf(stderr, "\n"); \
} while(0)

#define GGML_ANE_LOG_WARN(...) do { \
    fprintf(stderr, "[ANE WARN] " __VA_ARGS__); \
    fprintf(stderr, "\n"); \
} while(0)

#define GGML_ANE_LOG_ERROR(...) do { \
    fprintf(stderr, "[ANE ERROR] " __VA_ARGS__); \
    fprintf(stderr, "\n"); \
} while(0)

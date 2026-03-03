// ggml-ane.mm - ANE backend registration and main entry point
#import <Foundation/Foundation.h>
#include "ggml-ane.h"
#include "ggml-ane-impl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <new>
#include <map>
#include <mutex>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_CPU_ARM64
#define GGML_ANE_AVAILABLE 1
#else
#define GGML_ANE_AVAILABLE 0
#endif
#else
#define GGML_ANE_AVAILABLE 0
#endif

////////////////////////////////////////////////////////////////////////////////
// External declarations (from ggml-ane-device.mm)
////////////////////////////////////////////////////////////////////////////////

extern "C" struct ggml_backend_device g_ane_device;
extern "C" bool g_ane_device_initialized;
extern "C" bool ggml_ane_device_init(void);

////////////////////////////////////////////////////////////////////////////////
// Kernel Cache
////////////////////////////////////////////////////////////////////////////////

static std::map<uint64_t, ggml_ane_kernel_t> g_matmul_kernels;
static std::mutex g_matmul_kernels_mutex;

// Simple hash for matmul dimensions
static uint64_t hash_matmul_dims(int64_t m, int64_t n, int64_t k) {
    return ((uint64_t)m << 42) | ((uint64_t)n << 21) | (uint64_t)k;
}

////////////////////////////////////////////////////////////////////////////////
// GUID Definition
////////////////////////////////////////////////////////////////////////////////

static ggml_guid_t ggml_backend_ane_guid(void) {
    static ggml_guid guid = { 
        0x41, 0x4e, 0x45, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    return &guid;
}

////////////////////////////////////////////////////////////////////////////////
// Backend Interface
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_name(ggml_backend_t backend) {
    return "ANE";
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_free(ggml_backend_t backend) {
    ggml_ane_context * ctx = (ggml_ane_context *)backend->context;
    if (ctx) {
        delete ctx;
    }
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_ane_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_ane_buffer_type();
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);
    GGML_UNUSED(backend);
}

static bool ggml_backend_ane_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (src->buffer && dst->buffer) {
        return false;
    }
    return false;
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
}

static void ggml_backend_ane_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
}

static ggml_backend_graph_plan_t ggml_backend_ane_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    ggml_ane_graph_plan * plan = new ggml_ane_graph_plan();
    plan->graph = const_cast<struct ggml_cgraph *>(cgraph);
    plan->node_supported.resize(cgraph->n_nodes, false);
    plan->node_order.resize(cgraph->n_nodes);
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        plan->node_order[i] = i;
    }
    
    return (ggml_backend_graph_plan_t)plan;
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ggml_ane_graph_plan * ctx = (ggml_ane_graph_plan *)plan;
    if (ctx) {
        delete ctx;
    }
    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_ane_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    return GGML_STATUS_FAILED;
    GGML_UNUSED(backend);
    GGML_UNUSED(plan);
}

////////////////////////////////////////////////////////////////////////////////
// Graph Execution - Core Implementation
////////////////////////////////////////////////////////////////////////////////

// Execute a single MUL_MAT on ANE
static bool ggml_ane_exec_mul_mat(struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];  // Weights [K, N] or [N, K]
    struct ggml_tensor * src1 = dst->src[1];  // Input [K, M] or [M, K]
    
    if (!src0 || !src1) {
        return false;
    }
    
    // Get dimensions
    // src0 (weights): ne[0] x ne[1] = K x N (or transposed)
    // src1 (input): ne[0] x ne[1] = K x M (for batch M)
    // dst (output): ne[0] x ne[1] = N x M
    
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    
    // Handle transposed weights (common in llama.cpp)
    // If src0->ne[0] != src1->ne[0], weights might be transposed
    bool src0_transposed = false;
    if (src0->ne[0] != src1->ne[0]) {
        // Check if transposed makes sense
        if (src0->ne[1] == src1->ne[0]) {
            src0_transposed = true;
        }
    }
    
    // Adjust dimensions based on transpose
    int64_t in_ch, out_ch, spatial;
    if (src0_transposed) {
        // src0 is [N, K], need to treat as [K, N]
        in_ch = N;   // K in original terms
        out_ch = K;  // N in original terms
    } else {
        // src0 is [K, N]
        in_ch = K;
        out_ch = N;
    }
    spatial = M;
    
    // Skip if too small (not worth ANE overhead)
    if (in_ch * out_ch * spatial < 64 * 1024) {
        return false;  // < 64K elements, CPU is faster
    }
    
    // Skip if too large for ANE SRAM (~32MB working set)
    size_t working_set = in_ch * out_ch * 2 + in_ch * spatial * 2 + out_ch * spatial * 2;
    if (working_set > 64 * 1024 * 1024) {  // 64MB limit (allow some spill)
        GGML_ANE_LOG_DEBUG("MUL_MAT too large for ANE: %zu MB working set", working_set / (1024 * 1024));
        return false;
    }
    
    // Check if we have a cached kernel for these dimensions
    uint64_t hash = hash_matmul_dims(in_ch, out_ch, spatial);
    
    ggml_ane_kernel_t kernel = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_matmul_kernels_mutex);
        auto it = g_matmul_kernels.find(hash);
        if (it != g_matmul_kernels.end()) {
            kernel = it->second;
        }
    }
    
    // Compile kernel if not cached
    if (!kernel) {
        // Generate MIL for conv-based matmul
        NSString * mil = ggml_ane_gen_mil_conv(in_ch, out_ch, spatial);
        const char * mil_cstr = [mil UTF8String];
        
        // Build weight blob
        // Weights need to be [out_ch, in_ch] for conv format
        float * weights_transposed = nullptr;
        const float * weights = (const float *)src0->data;
        
        if (src0_transposed) {
            // Already [N, K] = [out_ch, in_ch], use directly
            weights_transposed = nullptr;
        } else {
            // Need to transpose from [K, N] to [N, K]
            weights_transposed = (float *)malloc(out_ch * in_ch * sizeof(float));
            for (int64_t n = 0; n < out_ch; n++) {
                for (int64_t k = 0; k < in_ch; k++) {
                    weights_transposed[n * in_ch + k] = weights[k * out_ch + n];
                }
            }
            weights = weights_transposed;
        }
        
        NSData * weight_blob = ggml_ane_build_weight_blob(weights, out_ch, in_ch);
        
        if (weights_transposed) {
            free(weights_transposed);
        }
        
        // Compile
        size_t input_size = in_ch * spatial * sizeof(float);
        size_t output_size = out_ch * spatial * sizeof(float);
        
        kernel = ggml_ane_compile_kernel(
            mil_cstr,
            [weight_blob bytes],
            [weight_blob length],
            1, &input_size,
            1, &output_size,
            hash  // Cache key
        );
        
        if (!kernel) {
            GGML_ANE_LOG_ERROR("Failed to compile ANE kernel for MUL_MAT");
            return false;
        }
        
        // Cache it
        {
            std::lock_guard<std::mutex> lock(g_matmul_kernels_mutex);
            g_matmul_kernels[hash] = kernel;
        }
    }
    
    // Prepare input data
    // Input is [K, M] in row-major, ANE expects [1, K, 1, M]
    float * input_conv = (float *)malloc(in_ch * spatial * sizeof(float));
    const float * src1_data = (const float *)src1->data;
    
    for (int64_t m = 0; m < spatial; m++) {
        for (int64_t k = 0; k < in_ch; k++) {
            // src1 is [K, M] row-major, ANE wants [K, M] but as [1, K, 1, M]
            input_conv[k * spatial + m] = src1_data[m * in_ch + k];
        }
    }
    
    // Allocate output
    float * output_conv = (float *)malloc(out_ch * spatial * sizeof(float));
    
    // Execute
    const void * inputs[1] = { input_conv };
    void * outputs[1] = { output_conv };
    
    bool success = ggml_ane_execute(kernel, inputs, outputs);
    
    if (!success) {
        GGML_ANE_LOG_ERROR("ANE execution failed for MUL_MAT");
        free(input_conv);
        free(output_conv);
        return false;
    }
    
    // Convert output back to row-major [N, M]
    float * dst_data = (float *)dst->data;
    for (int64_t m = 0; m < spatial; m++) {
        for (int64_t n = 0; n < out_ch; n++) {
            // ANE output is [1, N, 1, M], convert to [N, M] row-major
            dst_data[m * out_ch + n] = output_conv[n * spatial + m];
        }
    }
    
    free(input_conv);
    free(output_conv);
    
    return true;
}

static enum ggml_status ggml_backend_ane_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (!GGML_ANE_AVAILABLE) {
        GGML_ANE_LOG_DEBUG("ANE not available on this platform");
        return GGML_STATUS_FAILED;
    }
    
    // Initialize runtime if needed
    if (!ggml_ane_runtime_init()) {
        GGML_ANE_LOG_DEBUG("ANE runtime init failed");
        return GGML_STATUS_FAILED;
    }
    
    GGML_ANE_LOG_INFO("ANE graph compute: analyzing %d nodes...", cgraph->n_nodes);
    
    int mul_mat_ops = 0;
    int supported_ops = 0;
    int unsupported_ops = 0;
    
    // Analyze graph first
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        if (node->op == GGML_OP_NONE) {
            continue;  // Skip placeholder nodes
        }
        
        if (node->op == GGML_OP_MUL_MAT) {
            mul_mat_ops++;
            
            // Check why we might not support it
            if (!node->src[0] || !node->src[1]) {
                GGML_ANE_LOG_DEBUG("  MUL_MAT %d: missing src tensors", i);
                unsupported_ops++;
                continue;
            }
            
            // Check if quantized (ANE only supports FP16/FP32)
            if (node->src[0]->type != GGML_TYPE_F32 && node->src[0]->type != GGML_TYPE_F16) {
                GGML_ANE_LOG_DEBUG("  MUL_MAT %d: quantized weights (type=%d), skipping", i, node->src[0]->type);
                unsupported_ops++;
                continue;
            }
            
            if (node->src[1]->type != GGML_TYPE_F32 && node->src[1]->type != GGML_TYPE_F16) {
                GGML_ANE_LOG_DEBUG("  MUL_MAT %d: quantized input (type=%d), skipping", i, node->src[1]->type);
                unsupported_ops++;
                continue;
            }
            
            const int64_t ne0 = node->src[0]->ne[0];
            const int64_t ne1 = node->src[0]->ne[1];
            const int64_t M = node->src[1]->ne[1];
            
            // Skip small matrices
            if (ne0 * ne1 * M < 64 * 1024) {
                GGML_ANE_LOG_DEBUG("  MUL_MAT %d: too small (%ldx%ldx%ld), CPU faster", i, ne0, ne1, M);
                unsupported_ops++;
                continue;
            }
            
            // Check SRAM limit
            size_t working_set = ne0 * ne1 * 2 + ne0 * M * 2 + ne1 * M * 2;
            if (working_set > 64 * 1024 * 1024) {
                GGML_ANE_LOG_DEBUG("  MUL_MAT %d: too large (%zu MB)", i, working_set / (1024 * 1024));
                unsupported_ops++;
                continue;
            }
            
            GGML_ANE_LOG_DEBUG("  MUL_MAT %d: dimensions %ldx%ldx%ld, could use ANE", i, ne0, ne1, M);
            supported_ops++;
        } else {
            // For now, we only implement MUL_MAT
            // TODO: Implement ADD, MUL, SOFT_MAX, etc.
            const char * op_name = ggml_op_name(node->op);
            GGML_ANE_LOG_DEBUG("  Op %d: %s (type=%d), not yet implemented in ANE", 
                              i, op_name ? op_name : "unknown", node->op);
            unsupported_ops++;
        }
    }
    
    GGML_ANE_LOG_INFO("ANE graph analysis: %d MUL_MAT, %d could use ANE, %d unsupported", 
                      mul_mat_ops, supported_ops, unsupported_ops);
    
    // If we can't handle all ops, let ggml fall back to CPU/Metal
    if (unsupported_ops > 0) {
        GGML_ANE_LOG_INFO("ANE: falling back to CPU/Metal (%d unsupported ops)", unsupported_ops);
        return GGML_STATUS_FAILED;
    }
    
    // Process each node
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!ggml_ane_exec_mul_mat(node)) {
                    GGML_ANE_LOG_ERROR("ANE: MUL_MAT execution failed");
                    return GGML_STATUS_FAILED;
                }
                break;
                
            case GGML_OP_NONE:
                // Skip placeholder
                break;
                
            default:
                // Shouldn't reach here if analysis was correct
                GGML_ANE_LOG_ERROR("ANE: Unexpected op %d in execution phase", node->op);
                return GGML_STATUS_FAILED;
        }
    }
    
    GGML_ANE_LOG_INFO("ANE: graph compute complete");
    return GGML_STATUS_SUCCESS;
    
    GGML_UNUSED(backend);
}

static bool ggml_backend_ane_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    if (!GGML_ANE_AVAILABLE) {
        return false;
    }
    
    // For now, we only implement MUL_MAT
    // TODO: Implement ADD, MUL, SOFT_MAX, RMS_NORM
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            // Check dimensions
            if (!op->src[0] || !op->src[1]) return false;
            
            // Check if quantized (ANE only supports FP16/FP32)
            if (op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) return false;
            if (op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_F16) return false;
            
            const int64_t ne0 = op->src[0]->ne[0];
            const int64_t ne1 = op->src[0]->ne[1];
            const int64_t M = op->src[1]->ne[1];
            
            // Skip small matrices (CPU is faster)
            if (ne0 * ne1 * M < 64 * 1024) return false;
            
            // Check SRAM limit
            size_t working_set = ne0 * ne1 * 2 + ne0 * M * 2 + ne1 * M * 2;
            if (working_set > 64 * 1024 * 1024) return false;
            
            return true;
        }
        default:
            return false;
    }
    GGML_UNUSED(backend);
}

static bool ggml_backend_ane_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    if (!ggml_backend_ane_supports_op(backend, op)) {
        return false;
    }
    
    // Only offload MUL_MAT for now (implemented)
    if (op->op == GGML_OP_MUL_MAT) {
        return true;
    }
    
    return false;
}

static bool ggml_backend_ane_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // ANE works with CPU buffers (unified memory)
    return buft == ggml_backend_cpu_buffer_type() || 
           buft == ggml_backend_ane_buffer_type();
    GGML_UNUSED(backend);
}

static ggml_backend_dev_t ggml_backend_ane_get_device_wrapper(ggml_backend_t backend) {
    return ggml_backend_ane_get_device(0);
    GGML_UNUSED(backend);
}

////////////////////////////////////////////////////////////////////////////////

static ggml_backend_i ggml_backend_ane_interface = {
    /* .get_name                = */ ggml_backend_ane_name,
    /* .free                    = */ ggml_backend_ane_free,
    /* .set_tensor_async        = */ ggml_backend_ane_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_ane_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_ane_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_ane_synchronize,
    /* .graph_plan_create       = */ ggml_backend_ane_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_ane_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_ane_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_ane_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////

ggml_backend_t ggml_backend_ane_init(void) {
    if (!GGML_ANE_AVAILABLE) {
        GGML_ANE_LOG_WARN("ANE not available on this platform");
        return nullptr;
    }
    
    ggml_ane_device_init();
    
    ggml_ane_context * ctx = new ggml_ane_context();
    memset(ctx, 0, sizeof(*ctx));
    
    ggml_backend_t backend = new ggml_backend();
    backend->guid = ggml_backend_ane_guid();
    backend->iface = ggml_backend_ane_interface;
    backend->device = ggml_backend_ane_get_device(0);
    backend->context = ctx;
    
    GGML_ANE_LOG_INFO("ANE backend initialized");
    return backend;
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_ane_name;
}

////////////////////////////////////////////////////////////////////////////////
// Registration
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_reg_get_name(ggml_backend_reg_t reg) {
    return "ANE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_ane_reg_get_device_count(ggml_backend_reg_t reg) {
    return GGML_ANE_AVAILABLE ? 1 : 0;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_ane_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (!GGML_ANE_AVAILABLE || index != 0) {
        return nullptr;
    }
    
    if (!g_ane_device_initialized) {
        if (!ggml_ane_device_init()) {
            return nullptr;
        }
    }
    
    g_ane_device.reg = reg;
    return &g_ane_device;
}

static void * ggml_backend_ane_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_ane_init") == 0) {
        return (void *)ggml_backend_ane_init;
    }
    if (strcmp(name, "ggml_backend_ane_get_device") == 0) {
        return (void *)ggml_backend_ane_get_device;
    }
    return nullptr;
    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_ane_reg_interface = {
    /* .get_name          = */ ggml_backend_ane_reg_get_name,
    /* .get_device_count  = */ ggml_backend_ane_reg_get_device_count,
    /* .get_device        = */ ggml_backend_ane_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_ane_reg_get_proc_address,
};

static ggml_backend_reg ggml_backend_ane_reg_container = {
    /* .api_version  = */ GGML_BACKEND_API_VERSION,
    /* .iface        = */ ggml_backend_ane_reg_interface,
    /* .context      = */ nullptr,
};

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    return &ggml_backend_ane_reg_container;
}

////////////////////////////////////////////////////////////////////////////////
// Capability Queries
////////////////////////////////////////////////////////////////////////////////

bool ggml_ane_supports_op(const struct ggml_tensor * op) {
    return ggml_backend_ane_supports_op(nullptr, op);
}

bool ggml_ane_offload_op(const struct ggml_tensor * op) {
    return ggml_backend_ane_offload_op(nullptr, op);
}

////////////////////////////////////////////////////////////////////////////////
// Performance Stats
////////////////////////////////////////////////////////////////////////////////

void ggml_ane_get_perf_stats(struct ggml_ane_perf_stats * stats) {
    memset(stats, 0, sizeof(*stats));
}

void ggml_ane_reset_perf_stats(void) {
}

////////////////////////////////////////////////////////////////////////////////
// Dynamic Loading Support
////////////////////////////////////////////////////////////////////////////////

GGML_BACKEND_DL_IMPL(ggml_backend_ane_reg)

// ggml-ane.mm - ANE backend registration and main entry point
#import <Foundation/Foundation.h>
#include "ggml-ane.h"
#include "ggml-ane-impl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
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
// Simple Op Execution (CPU-based, these are memory-bound anyway)
////////////////////////////////////////////////////////////////////////////////

static void ggml_ane_exec_add(struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];
    
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    const float * b = (const float *)src1->data;
    float * c = (float *)dst->data;
    
    // Simple vectorized add
    #pragma omp parallel for
    for (int64_t i = 0; i < ne; i++) {
        c[i] = a[i] + b[i];
    }
}

static void ggml_ane_exec_mul(struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];
    
    const int64_t ne = ggml_nelements(dst);
    const float * a = (const float *)src0->data;
    const float * b = (const float *)src1->data;
    float * c = (float *)dst->data;
    
    // Simple vectorized mul
    #pragma omp parallel for
    for (int64_t i = 0; i < ne; i++) {
        c[i] = a[i] * b[i];
    }
}

static void ggml_ane_exec_rms_norm(struct ggml_tensor * dst) {
    struct ggml_tensor * src = dst->src[0];
    
    const int64_t ne00 = src->ne[0];
    const int64_t ne01 = src->ne[1];
    const int64_t ne02 = src->ne[2];
    const int64_t ne03 = src->ne[3];
    
    const float * x = (const float *)src->data;
    float * y = (float *)dst->data;
    
    // RMS norm: y = x * rsqrt(mean(x^2) + eps)
    const float eps = 1e-5f;  // Standard epsilon
    
    // For each row, compute RMS and normalize
    const int64_t n_rows = ne01 * ne02 * ne03;
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n_rows; i++) {
        const float * xi = x + i * ne00;
        float * yi = y + i * ne00;
        
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < ne00; j++) {
            sum_sq += xi[j] * xi[j];
        }
        
        // Compute RMS
        float rms = sqrtf(sum_sq / ne00 + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize
        for (int64_t j = 0; j < ne00; j++) {
            yi[j] = xi[j] * inv_rms;
        }
    }
}

static void ggml_ane_exec_softmax(struct ggml_tensor * dst) {
    struct ggml_tensor * src = dst->src[0];
    
    const int64_t ne00 = src->ne[0];
    const int64_t ne01 = src->ne[1];
    const int64_t ne02 = src->ne[2];
    const int64_t ne03 = src->ne[3];
    
    const float * x = (const float *)src->data;
    float * y = (float *)dst->data;
    
    // Softmax per row
    const int64_t n_rows = ne01 * ne02 * ne03;
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n_rows; i++) {
        const float * xi = x + i * ne00;
        float * yi = y + i * ne00;
        
        // Find max for numerical stability
        float max_val = xi[0];
        for (int64_t j = 1; j < ne00; j++) {
            if (xi[j] > max_val) max_val = xi[j];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < ne00; j++) {
            yi[j] = expf(xi[j] - max_val);
            sum += yi[j];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < ne00; j++) {
            yi[j] *= inv_sum;
        }
    }
}

static bool ggml_ane_exec_simple_op(struct ggml_tensor * node) {
    // Simple elementwise ops are memory-bound, so CPU is fine
    // The value of ANE is for compute-bound ops like MUL_MAT
    
    const char * op_name = ggml_op_name(node->op);
    GGML_ANE_LOG_DEBUG("  Executing %s on CPU (memory-bound)", op_name);
    
    // For these ops, we just pass through - ggml has already computed them
    // or they will be computed by the CPU backend
    return true;
}

static bool ggml_ane_exec_view_op(struct ggml_tensor * node) {
    // VIEW, RESHAPE, PERMUTE, etc. are just metadata ops
    // No actual computation needed
    GGML_ANE_LOG_DEBUG("  VIEW/RESHAPE op: no compute needed");
    return true;
}

// Execute a single MUL_MAT on ANE
static bool ggml_ane_exec_mul_mat(struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];  // Weights [K, N] or [N, K]
    struct ggml_tensor * src1 = dst->src[1];  // Input [K, M] or [M, K]
    
    if (!src0 || !src1) {
        return false;
    }
    
    // Get dimensions
    // In ggml MUL_MAT: dst = src0^T @ src1
    // src0: ne[0] x ne[1] x ne[2] x ne[3]
    // src1: ne[0] x ne[1] x ne[2] x ne[3]
    // dst:  ne[0] x ne[1] x ne[2] x ne[3]
    
    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // N
    const int64_t ne10 = src1->ne[0];  // K (should match ne00)
    const int64_t ne11 = src1->ne[1];  // M (batch size)
    const int64_t ne20 = dst->ne[0];   // N
    const int64_t ne21 = dst->ne[1];   // M
    
    GGML_ANE_LOG_DEBUG("MUL_MAT dims: src0[%ld,%ld], src1[%ld,%ld], dst[%ld,%ld]",
                       ne00, ne01, ne10, ne11, ne20, ne21);
    
    // For conv-based matmul:
    // Input [K, M] -> [1, K, 1, M]
    // Weights [N, K] -> [N, K, 1, 1]
    // Output [N, M] -> [1, N, 1, M]
    
    const int64_t K = ne00;  // Input channels
    const int64_t N = ne01;  // Output channels
    const int64_t M = ne11;  // Spatial (batch)
    
    // ANE requires minimum spatial dimension of 16
    // Pad to at least 16 to avoid "Program Inference error"
    const int64_t M_padded = (M < 16) ? 16 : M;
    
    GGML_ANE_LOG_DEBUG("Conv dims: in_ch=%ld, out_ch=%ld, spatial=%ld (padded=%ld)", 
                       K, N, M, M_padded);
    
    // Skip if too small (not worth ANE overhead)
    if (K * N * M < 64 * 1024) {
        return false;  // < 64K elements, CPU is faster
    }
    
    // Skip if too large for ANE SRAM
    // ANE has ~16MB SRAM per core, but can spill to DRAM
    // Allow up to 256MB for larger models (with DRAM spill)
    size_t working_set = K * N * 2 + K * M_padded * 2 + N * M_padded * 2;
    if (working_set > 256 * 1024 * 1024) {  // 256MB limit
        GGML_ANE_LOG_DEBUG("MUL_MAT too large for ANE: %zu MB working set", working_set / (1024 * 1024));
        return false;
    }
    
    // Check if we have a cached kernel for these dimensions
    // Use padded dimensions for hash to ensure cache hit for same padded size
    uint64_t hash = hash_matmul_dims(K, N, M_padded);
    
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
        // Transpose weights for weight blob
        // Weights in src0 are [K, N] with strides nb[0], nb[1]
        // Element (k, n) at k*nb[0] + n*nb[1]
        // Need to transpose to [N, K] for conv format
        float * weights_transposed = (float *)malloc(N * K * sizeof(float));
        
        GGML_ANE_LOG_DEBUG("src0 type: %d (F16=%d, F32=%d)", src0->type, GGML_TYPE_F16, GGML_TYPE_F32);
        GGML_ANE_LOG_DEBUG("src0: nb[0]=%ld, nb[1]=%ld, K=%ld, N=%ld", 
                           src0->nb[0], src0->nb[1], K, N);
        
        if (src0->type == GGML_TYPE_F16) {
            const ggml_fp16_t * weights_f16 = (const ggml_fp16_t *)src0->data;
            GGML_ANE_LOG_DEBUG("src0 is FP16, converting to FP32 during transpose");
            
            for (int64_t n = 0; n < N; n++) {
                for (int64_t k = 0; k < K; k++) {
                    // src0[K, N]: element (k, n) at k*nb[0] + n*nb[1]
                    const ggml_fp16_t * w_ptr = (const ggml_fp16_t *)((const char *)weights_f16 + k * src0->nb[0] + n * src0->nb[1]);
                    weights_transposed[n * K + k] = ggml_fp16_to_fp32(*w_ptr);
                }
            }
        } else {
            const float * weights_f32 = (const float *)src0->data;
            GGML_ANE_LOG_DEBUG("src0 is FP32");
            
            for (int64_t n = 0; n < N; n++) {
                for (int64_t k = 0; k < K; k++) {
                    // src0[K, N]: element (k, n) at k*nb[0] + n*nb[1]
                    const float * w_ptr = (const float *)((const char *)weights_f32 + k * src0->nb[0] + n * src0->nb[1]);
                    weights_transposed[n * K + k] = *w_ptr;
                }
            }
        }
        
        GGML_ANE_LOG_DEBUG("First 5 transposed weights: %.6f %.6f %.6f %.6f %.6f",
                           weights_transposed[0], weights_transposed[1], weights_transposed[2],
                           weights_transposed[3], weights_transposed[4]);
        
        // Generate MIL for conv-based matmul
        // Input: [1, K, 1, M_padded], Weights: [N, K, 1, 1], Output: [1, N, 1, M_padded]
        NSString * mil = ggml_ane_gen_mil_conv(K, N, M_padded);
        const char * mil_cstr = [mil UTF8String];
        
        GGML_ANE_LOG_DEBUG("MIL generated for K=%ld, N=%ld, M=%ld (using conv format)", K, N, M_padded);
        GGML_ANE_LOG_DEBUG("MIL text:\n%s", mil_cstr);
        
        NSData * weight_blob = ggml_ane_build_weight_blob(weights_transposed, N, K);
        free(weights_transposed);
        
        GGML_ANE_LOG_DEBUG("Weight blob size: %zu bytes", [weight_blob length]);
        
        // Compile
        size_t input_size = K * M_padded * sizeof(float);
        size_t output_size = N * M_padded * sizeof(float);
        
        GGML_ANE_LOG_DEBUG("Buffer sizes: input=%zu, output=%zu", input_size, output_size);
        
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
    
    // Prepare input data with padding
    // src1 has ne[0]=K, ne[1]=M
    // Element (k, m) at k*nb[0] + m*nb[1]
    // ANE expects [1, K, 1, M_padded] NCHW: element (k, m) at k*M_padded + m
    // Need to transpose and zero-pad for m >= M
    
    GGML_ANE_LOG_DEBUG("src1 type: %d (F16=%d, F32=%d)", src1->type, GGML_TYPE_F16, GGML_TYPE_F32);
    GGML_ANE_LOG_DEBUG("src1: ne[0]=%ld, ne[1]=%ld, nb[0]=%ld, nb[1]=%ld",
                       src1->ne[0], src1->ne[1], src1->nb[0], src1->nb[1]);
    
    float * input_x = (float *)calloc(K * M_padded, sizeof(float));  // Zero-initialized for padding
    
    if (src1->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src1_f16 = (const ggml_fp16_t *)src1->data;
        GGML_ANE_LOG_DEBUG("src1 is FP16, converting and transposing");
        
        // Transpose from ggml [K, M] to ANE [K, M_padded]
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < M; m++) {
                // src1[K, M]: element (k, m) at k*nb[0] + m*nb[1]
                const ggml_fp16_t * in_ptr = (const ggml_fp16_t *)((const char *)src1_f16 + k * src1->nb[0] + m * src1->nb[1]);
                // ANE: element (k, m) at k*M_padded + m
                input_x[k * M_padded + m] = ggml_fp16_to_fp32(*in_ptr);
            }
        }
    } else {
        const float * src1_f32 = (const float *)src1->data;
        GGML_ANE_LOG_DEBUG("src1 is FP32, transposing");
        
        // Transpose from ggml [K, M] to ANE [K, M_padded]
        for (int64_t k = 0; k < K; k++) {
            for (int64_t m = 0; m < M; m++) {
                // src1[K, M]: element (k, m) at k*nb[0] + m*nb[1]
                size_t byte_offset = k * src1->nb[0] + m * src1->nb[1];
                const float * in_ptr = (const float *)((const char *)src1_f32 + byte_offset);
                // ANE: element (k, m) at k*M_padded + m
                input_x[k * M_padded + m] = *in_ptr;
            }
        }
    }
    
    // Debug: print first few input values
    GGML_ANE_LOG_DEBUG("First 5 input values: %.6f %.6f %.6f %.6f %.6f",
                       input_x[0], input_x[1], input_x[2],
                       input_x[3], input_x[4]);
    
    // Allocate output (padded size)
    float * output = (float *)malloc(N * M_padded * sizeof(float));
    
    // Execute with 1 input (weights are embedded in kernel)
    const void * inputs[1] = { input_x };
    void * outputs[1] = { output };
    
    bool success = ggml_ane_execute(kernel, inputs, outputs);
    
    if (!success) {
        GGML_ANE_LOG_ERROR("ANE execution failed for MUL_MAT");
        free(input_x);
        free(output);
        return false;
    }
    
    // Copy output to dst with transpose (only valid M elements, skip padding)
    // ANE output is [1, N, 1, M_padded] NCHW: element (n, m) at n*M_padded + m
    // ggml dst has ne[0]=N, ne[1]=M: element (n, m) at n*nb[0] + m*nb[1]
    // These are TRANSPOSED! Need to swap indices.
    float * dst_data = (float *)dst->data;
    
    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            // ANE: output[n*M_padded + m]
            // dst[N, M]: element (n, m) at n*nb[0] + m*nb[1]
            float * dst_ptr = (float *)((char *)dst_data + n * dst->nb[0] + m * dst->nb[1]);
            *dst_ptr = output[n * M_padded + m];
        }
    }
    
    free(input_x);
    free(output);
    
    GGML_ANE_LOG_DEBUG("MUL_MAT completed successfully");
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
            
            // Check SRAM limit (allow up to 256MB with DRAM spill)
            size_t working_set = ne0 * ne1 * 2 + ne0 * M * 2 + ne1 * M * 2;
            if (working_set > 256 * 1024 * 1024) {
                GGML_ANE_LOG_DEBUG("  MUL_MAT %d: too large (%zu MB)", i, working_set / (1024 * 1024));
                unsupported_ops++;
                continue;
            }
            
            GGML_ANE_LOG_DEBUG("  MUL_MAT %d: dimensions %ldx%ldx%ld, could use ANE", i, ne0, ne1, M);
            supported_ops++;
        } else if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL || 
                   node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_SOFT_MAX) {
            // These ops are implemented (CPU execution within ANE backend)
            const char * op_name = ggml_op_name(node->op);
            GGML_ANE_LOG_DEBUG("  %s %d: supported (CPU execution)", op_name ? op_name : "unknown", i);
            supported_ops++;
        } else if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                   node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE ||
                   node->op == GGML_OP_CONT || node->op == GGML_OP_CPY ||
                   node->op == GGML_OP_NONE) {
            // Metadata ops - no compute needed
            supported_ops++;
        } else {
            // Unsupported op
            const char * op_name = ggml_op_name(node->op);
            GGML_ANE_LOG_DEBUG("  Op %d: %s (type=%d), not supported by ANE", 
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
            
            case GGML_OP_ADD:
                ggml_ane_exec_add(node);
                break;
            
            case GGML_OP_MUL:
                ggml_ane_exec_mul(node);
                break;
            
            case GGML_OP_RMS_NORM:
                ggml_ane_exec_rms_norm(node);
                break;
            
            case GGML_OP_SOFT_MAX:
                ggml_ane_exec_softmax(node);
                break;
            
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
            case GGML_OP_CPY:
                // No-op metadata ops
                break;
                
            case GGML_OP_NONE:
                // Skip placeholder
                break;
                
            default:
                // Shouldn't reach here if analysis was correct
                const char * op_name = ggml_op_name(node->op);
                GGML_ANE_LOG_ERROR("ANE: Unexpected op %s (%d) in execution phase", 
                                   op_name ? op_name : "unknown", node->op);
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
            
            // Check SRAM limit (allow up to 256MB with DRAM spill)
            size_t working_set = ne0 * ne1 * 2 + ne0 * M * 2 + ne1 * M * 2;
            if (working_set > 256 * 1024 * 1024) return false;
            
            return true;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
            return true;  // Metadata ops
        default:
            return false;
    }
    GGML_UNUSED(backend);
}

static bool ggml_backend_ane_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    if (!ggml_backend_ane_supports_op(backend, op)) {
        return false;
    }
    
    // Offload all supported ops
    return true;
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

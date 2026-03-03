// ggml-ane-ops.cpp - Operation mapping and execution
// Phase 4: Map ggml ops to ANE kernels

#include "ggml-ane-impl.h"
#include "ggml-ane.h"

#include <string.h>

// Determine if an operation should use ANE or CPU fallback
bool ggml_ane_should_offload_op(const struct ggml_tensor * op) {
    if (!op) return false;
    
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            // Check size constraints
            const int64_t M = op->src[1]->ne[1];
            const int64_t K = op->src[0]->ne[0];
            const int64_t N = op->src[0]->ne[1];
            
            // Avoid dispatch-limited small ops (< 0.1ms compute)
            // ANE dispatch overhead is ~0.095ms
            int64_t flops = 2 * M * N * K;
            if (flops < 1000000) { // < 1M FLOPs
                return false;
            }
            
            // Check SRAM limit (~32MB working set)
            size_t working_set = (M * K + K * N + M * N) * 2; // FP16
            if (working_set > 64 * 1024 * 1024) {
                // Might still benefit, but watch for SRAM spill
                GGML_ANE_LOG_DEBUG("Large working set: %zu MB", working_set / (1024 * 1024));
            }
            
            return true;
        }
        
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            // Elementwise ops - check if tensors are already on ANE
            // If inputs are ANE buffers, do the op there to avoid transfer
            return false; // For now, CPU is faster for elementwise
        
        case GGML_OP_RMS_NORM:
            // Can be done on ANE but needs custom kernel
            return false; // Phase 3
        
        case GGML_OP_SOFT_MAX:
            // Attention softmax
            return false; // Phase 3
        
        case GGML_OP_SILU:
            // FFN activation
            return false; // Phase 3
        
        default:
            return false;
    }
}

// Get the optimal kernel type for an operation
enum ggml_ane_kernel_type {
    GGML_ANE_KERNEL_MATMUL,
    GGML_ANE_KERNEL_CONV,      // 1x1 conv (faster than matmul)
    GGML_ANE_KERNEL_ADD,
    GGML_ANE_KERNEL_MUL,
    GGML_ANE_KERNEL_RMS_NORM,
    GGML_ANE_KERNEL_SOFTMAX,
    GGML_ANE_KERNEL_SILU,
    GGML_ANE_KERNEL_FUSED_QKV, // Fused Q, K, V projections
    GGML_ANE_KERNEL_FUSED_FFN, // Fused FFN up (w1 + w3)
};

enum ggml_ane_kernel_type ggml_ane_get_kernel_type(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // Use conv for better performance
            return GGML_ANE_KERNEL_CONV;
        case GGML_OP_ADD:
            return GGML_ANE_KERNEL_ADD;
        case GGML_OP_MUL:
            return GGML_ANE_KERNEL_MUL;
        case GGML_OP_RMS_NORM:
            return GGML_ANE_KERNEL_RMS_NORM;
        case GGML_OP_SOFT_MAX:
            return GGML_ANE_KERNEL_SOFTMAX;
        case GGML_OP_SILU:
            return GGML_ANE_KERNEL_SILU;
        default:
            return GGML_ANE_KERNEL_MATMUL; // Fallback
    }
}

// Calculate input/output sizes for kernel compilation
void ggml_ane_calc_io_sizes(
    const struct ggml_tensor * op,
    size_t * input_sizes,
    size_t * output_sizes,
    int * n_inputs,
    int * n_outputs
) {
    *n_inputs = 0;
    *n_outputs = 0;
    
    switch (op->op) {
        case GGML_OP_MUL_MAT: {
            // Input: x[1, K, M], W[out_ch, in_ch, 1, 1] (baked)
            // Output: y[1, N, M]
            int64_t M = op->src[1]->ne[1];
            int64_t K = op->src[0]->ne[0];
            int64_t N = op->src[0]->ne[1];
            
            input_sizes[0] = M * K * 2; // FP16
            output_sizes[0] = M * N * 2;
            *n_inputs = 1;
            *n_outputs = 1;
            break;
        }
        
        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            int64_t ne = ggml_nelements(op);
            input_sizes[0] = ne * 2;
            input_sizes[1] = ne * 2;
            output_sizes[0] = ne * 2;
            *n_inputs = 2;
            *n_outputs = 1;
            break;
        }
        
        case GGML_OP_SILU: {
            int64_t ne = ggml_nelements(op);
            input_sizes[0] = ne * 2;
            output_sizes[0] = ne * 2;
            *n_inputs = 1;
            *n_outputs = 1;
            break;
        }
        
        default:
            break;
    }
}

// Transpose tensor data for ANE format
// ANE expects [1, channels, spatial] format (channel-first)
// ggml uses row-major [spatial, channels]
void ggml_ane_transpose_to_ane_format(
    const float * src,
    float * dst,
    int64_t rows,
    int64_t cols
) {
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void ggml_ane_transpose_from_ane_format(
    const float * src,
    float * dst,
    int64_t channels,
    int64_t spatial
) {
    for (int64_t c = 0; c < channels; c++) {
        for (int64_t s = 0; s < spatial; s++) {
            dst[s * channels + c] = src[c * spatial + s];
        }
    }
}

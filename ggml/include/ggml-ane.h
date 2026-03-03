// ggml-ane.h - Apple Neural Engine backend for ggml
//
// This backend provides direct access to Apple's Neural Engine (ANE) via
// private _ANEClient APIs, bypassing CoreML for 2-4x lower overhead.
//
// Based on reverse engineering by maderix: https://github.com/maderix/ANE
//
// Requirements:
//   - macOS 15.0+ on Apple Silicon (M1-M4)
//   - ANE is a 16-core graph execution engine optimized for inference
//   - 19 TFLOPS FP16 at 2.8W (6.6 TFLOPS/W efficiency)
//
// Usage:
//   ggml_backend_reg_t reg = ggml_backend_ane_reg();
//   ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
//   ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
//
// Then use backend with standard ggml backend API.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Backend API
//

// Get the ANE backend registration
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ane_reg(void);

// Initialize ANE backend directly (alternative to going through reg)
GGML_BACKEND_API ggml_backend_t ggml_backend_ane_init(void);

// Check if backend is ANE
GGML_BACKEND_API bool ggml_backend_is_ane(ggml_backend_t backend);

//
// Device API
//

// Get ANE device count (always 1 on Apple Silicon, 0 otherwise)
GGML_BACKEND_API int ggml_backend_ane_device_count(void);

// Get ANE device by index
GGML_BACKEND_API ggml_backend_dev_t ggml_backend_ane_get_device(int index);

//
// Buffer API
//

// Get ANE buffer type (IOSurface-backed)
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void);

// Check if buffer is ANE buffer (IOSurface-backed)
GGML_BACKEND_API bool ggml_backend_buffer_is_ane(ggml_backend_buffer_t buffer);

//
// Capability Queries
//

// Check if ANE supports a specific operation
GGML_BACKEND_API bool ggml_ane_supports_op(const struct ggml_tensor * op);

// Check if ANE should offload an operation (vs CPU)
GGML_BACKEND_API bool ggml_ane_offload_op(const struct ggml_tensor * op);

//
// Performance Stats
//

struct ggml_ane_perf_stats {
    size_t n_kernels_compiled;    // Number of MIL kernels compiled
    size_t n_kernels_executed;    // Number of kernel executions
    double compile_time_ms;       // Total time spent compiling
    double exec_time_ms;          // Total time spent executing
    double power_watts;           // Estimated power consumption
    double tflops;                // Achieved TFLOPS
};

// Get performance statistics
GGML_BACKEND_API void ggml_ane_get_perf_stats(struct ggml_ane_perf_stats * stats);

// Reset performance statistics
GGML_BACKEND_API void ggml_ane_reset_perf_stats(void);

#ifdef __cplusplus
}
#endif

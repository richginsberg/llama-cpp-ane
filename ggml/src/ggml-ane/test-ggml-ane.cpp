// test-ggml-ane.cpp - Unit tests for ANE backend
#include "ggml-ane.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define TEST_PASSED(name) printf("[PASS] %s\n", name)
#define TEST_FAILED(name, reason) printf("[FAIL] %s: %s\n", name, reason)

static int tests_passed = 0;
static int tests_failed = 0;

static void test_backend_registration(void) {
    const char * name = "Backend Registration";
    
    ggml_backend_reg_t reg = ggml_backend_ane_reg();
    if (!reg) {
        TEST_FAILED(name, "ggml_backend_ane_reg() returned NULL");
        tests_failed++;
        return;
    }
    
    const char * reg_name = reg->iface.get_name(reg);
    if (!reg_name || strcmp(reg_name, "ANE") != 0) {
        TEST_FAILED(name, "Registration name is not 'ANE'");
        tests_failed++;
        return;
    }
    
    TEST_PASSED(name);
    tests_passed++;
}

static void test_device_count(void) {
    const char * name = "Device Count";
    
    ggml_backend_reg_t reg = ggml_backend_ane_reg();
    size_t count = reg->iface.get_device_count(reg);
    
    #ifdef __APPLE__
    #if TARGET_CPU_ARM64
    if (count != 1) {
        TEST_FAILED(name, "Expected 1 device on Apple Silicon");
        tests_failed++;
        return;
    }
    #else
    if (count != 0) {
        TEST_FAILED(name, "Expected 0 devices on non-ARM64");
        tests_failed++;
        return;
    }
    #endif
    #else
    if (count != 0) {
        TEST_FAILED(name, "Expected 0 devices on non-Apple");
        tests_failed++;
        return;
    }
    #endif
    
    TEST_PASSED(name);
    tests_passed++;
}

static void test_backend_init(void) {
    const char * name = "Backend Initialization";
    
    #if defined(__APPLE__) && TARGET_CPU_ARM64
    ggml_backend_t backend = ggml_backend_ane_init();
    if (!backend) {
        TEST_FAILED(name, "ggml_backend_ane_init() returned NULL");
        tests_failed++;
        return;
    }
    
    if (!ggml_backend_is_ane(backend)) {
        TEST_FAILED(name, "ggml_backend_is_ane() returned false");
        ggml_backend_free(backend);
        tests_failed++;
        return;
    }
    
    const char * backend_name = ggml_backend_name(backend);
    if (!backend_name || strcmp(backend_name, "ANE") != 0) {
        TEST_FAILED(name, "Backend name is not 'ANE'");
        ggml_backend_free(backend);
        tests_failed++;
        return;
    }
    
    ggml_backend_free(backend);
    TEST_PASSED(name);
    tests_passed++;
    #else
    printf("[SKIP] %s (not on Apple Silicon)\n", name);
    #endif
}

static void test_op_support_detection(void) {
    const char * name = "Operation Support Detection";
    
    #if defined(__APPLE__) && TARGET_CPU_ARM64
    // Initialize ggml
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        TEST_FAILED(name, "Failed to initialize ggml context");
        tests_failed++;
        return;
    }
    
    // Test MUL_MAT support
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    struct ggml_tensor * mul_mat = ggml_mul_mat(ctx, a, b);
    
    if (!ggml_ane_supports_op(mul_mat)) {
        TEST_FAILED(name, "MUL_MAT should be supported");
        tests_failed++;
    } else {
        tests_passed++;
    }
    
    // Test ROPE (should NOT be supported)
    struct ggml_tensor * rope = ggml_rope(ctx, a, b, 0, 128, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32, 1);
    if (ggml_ane_supports_op(rope)) {
        TEST_FAILED(name, "ROPE should NOT be supported");
        tests_failed++;
    }
    
    ggml_free(ctx);
    TEST_PASSED(name);
    #else
    printf("[SKIP] %s (not on Apple Silicon)\n", name);
    #endif
}

static void test_perf_stats(void) {
    const char * name = "Performance Stats";
    
    struct ggml_ane_perf_stats stats;
    ggml_ane_get_perf_stats(&stats);
    
    // Initial stats should be zeroed
    if (stats.n_kernels_compiled != 0 || stats.n_kernels_executed != 0) {
        TEST_FAILED(name, "Initial stats not zeroed");
        tests_failed++;
        return;
    }
    
    // Reset should work
    ggml_ane_reset_perf_stats();
    ggml_ane_get_perf_stats(&stats);
    
    if (stats.n_kernels_compiled != 0 || stats.n_kernels_executed != 0) {
        TEST_FAILED(name, "Stats not zeroed after reset");
        tests_failed++;
        return;
    }
    
    TEST_PASSED(name);
    tests_passed++;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    
    printf("=================================\n");
    printf("ggml-ane Backend Tests\n");
    printf("=================================\n\n");
    
    test_backend_registration();
    test_device_count();
    test_backend_init();
    test_op_support_detection();
    test_perf_stats();
    
    printf("\n=================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("=================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}

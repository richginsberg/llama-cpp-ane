// test-ane-matmul.cpp - Simple test to run a single matmul through ANE
// This tests the core ANE pipeline: MIL generation → compilation → execution

#include "ggml-ane.h"
#include "ggml-ane-impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_CPU_ARM64
#define TEST_ANE_AVAILABLE 1
#else
#define TEST_ANE_AVAILABLE 0
#endif
#else
#define TEST_ANE_AVAILABLE 0
#endif

// Test matrix dimensions
// Using sizes that are large enough to benefit from ANE
#define M 64      // Batch size / sequence length
#define K 256     // Input dimension
#define N 128     // Output dimension

static void init_matrix(float * mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
}

static void cpu_matmul(const float * A, const float * B, float * C, int M, int K, int N) {
    // C = A @ B where A is [M, K] and B is [K, N]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

static bool compare_matrices(const float * a, const float * b, int size, float tolerance = 1e-2f) {
    float max_diff = 0.0f;
    int mismatches = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (mismatches < 5) {
                printf("  Mismatch at %d: expected %.4f, got %.4f (diff %.4f)\n", 
                       i, a[i], b[i], diff);
            }
            mismatches++;
        }
    }
    
    printf("  Max diff: %.6f, Mismatches: %d / %d\n", max_diff, mismatches, size);
    return mismatches == 0;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    
    printf("=================================\n");
    printf("ANE Matmul Test\n");
    printf("=================================\n\n");
    
#if TEST_ANE_AVAILABLE
    
    // Step 1: Initialize ANE runtime
    printf("[1/5] Initializing ANE runtime...\n");
    if (!ggml_ane_runtime_init()) {
        printf("  FAILED: Could not initialize ANE runtime\n");
        return 1;
    }
    printf("  ✓ ANE runtime initialized\n\n");
    
    // Step 2: Generate MIL for matmul
    printf("[2/5] Generating MIL for matmul (%dx%d @ %dx%d)...\n", M, K, K, N);
    
    // Use conv-based matmul (3x faster on ANE)
    // MIL expects: input [1, K, M], weights [N, K, 1, 1], output [1, N, M]
    char mil_text[4096];
    snprintf(mil_text, sizeof(mil_text),
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(weight = W, x = x, "
        "dilations = [1, 1], groups = 1, pad = [0, 0, 0, 0], pad_type = \"valid\", "
        "strides = [1, 1])[name = string(\"conv\")];\n"
        "    } -> (y);\n"
        "}\n",
        K, M, N, K, N, K, N, M
    );
    
    printf("  MIL generated (%zu bytes)\n\n", strlen(mil_text));
    
    // Step 3: Create test data
    printf("[3/5] Creating test data...\n");
    
    // Allocate matrices (row-major)
    float * A = (float *)malloc(M * K * sizeof(float));  // Input [M, K]
    float * B = (float *)malloc(K * N * sizeof(float));  // Weights [K, N]
    float * C_cpu = (float *)malloc(M * N * sizeof(float));  // CPU result [M, N]
    float * C_ane = (float *)malloc(M * N * sizeof(float));  // ANE result [M, N]
    
    init_matrix(A, M, K, 0.1f);
    init_matrix(B, K, N, 0.1f);
    
    printf("  Input A: [%d, %d]\n", M, K);
    printf("  Weights B: [%d, %d]\n", K, N);
    printf("  Output C: [%d, %d]\n\n", M, N);
    
    // Compute CPU reference
    printf("[4/5] Computing CPU reference...\n");
    cpu_matmul(A, B, C_cpu, M, K, N);
    printf("  ✓ CPU matmul complete\n\n");
    
    // Step 5: Build weight blob for ANE
    // ANE expects weights as FP16 in [N, K, 1, 1] format
    printf("[5/5] Building weight blob for ANE...\n");
    
    size_t weight_size = N * K * sizeof(uint16_t);  // FP16
    size_t header_size = 64 + 64;  // Global header + chunk header
    size_t blob_size = header_size + weight_size;
    
    uint8_t * weight_blob = (uint8_t *)calloc(blob_size, 1);
    
    // Global header
    weight_blob[0] = 0x01;  // version
    weight_blob[4] = 0x02;  // subversion
    
    // Chunk header
    uint8_t * chunk = weight_blob + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;  // Magic
    chunk[4] = 0x01;  // version
    *(uint32_t *)(chunk + 8) = (uint32_t)weight_size;   // data_size
    *(uint32_t *)(chunk + 16) = 128;                    // data_offset
    
    // Convert B from [K, N] to [N, K] FP16
    // Weights need to be transposed for conv format
    uint16_t * fp16_weights = (uint16_t *)(weight_blob + 128);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            // Convert float to FP16 (simple truncation for test)
            float val = B[k * N + n];
            // IEEE FP32 to FP16 conversion (simplified)
            uint32_t bits = *(uint32_t *)&val;
            uint16_t fp16 = ((bits >> 16) & 0x8000) |  // sign
                           ((((bits >> 23) & 0xFF) - 127 + 15) << 10) |  // exponent
                           ((bits >> 13) & 0x3FF);  // mantissa
            fp16_weights[n * K + k] = fp16;
        }
    }
    
    printf("  Weight blob: %zu bytes\n\n", blob_size);
    
    // Compile kernel
    printf("[6/7] Compiling ANE kernel...\n");
    
    size_t input_size = K * M * sizeof(uint16_t);   // FP16
    size_t output_size = N * M * sizeof(uint16_t);  // FP16
    
    ggml_ane_kernel_t kernel = ggml_ane_compile_kernel(
        mil_text,
        weight_blob,
        blob_size,
        1, &input_size,
        1, &output_size,
        0  // No caching for test
    );
    
    if (!kernel) {
        printf("  FAILED: Could not compile kernel\n");
        free(A); free(B); free(C_cpu); free(C_ane); free(weight_blob);
        return 1;
    }
    printf("  ✓ Kernel compiled\n\n");
    
    // Prepare input (transpose A from [M, K] to [1, K, 1, M] for conv)
    printf("[7/7] Executing on ANE...\n");
    
    uint16_t * input_fp16 = (uint16_t *)malloc(input_size);
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float val = A[m * K + k];
            uint32_t bits = *(uint32_t *)&val;
            uint16_t fp16 = ((bits >> 16) & 0x8000) |
                           ((((bits >> 23) & 0xFF) - 127 + 15) << 10) |
                           ((bits >> 13) & 0x3FF);
            // ANE expects [1, K, 1, M] format
            input_fp16[k * M + m] = fp16;
        }
    }
    
    // Execute
    const void * inputs[1] = { input_fp16 };
    void * outputs[1] = { C_ane };
    
    // Note: Output will be FP16, need to allocate proper buffer
    uint16_t * output_fp16 = (uint16_t *)malloc(output_size);
    outputs[0] = output_fp16;
    
    bool success = ggml_ane_execute(kernel, inputs, outputs);
    
    if (!success) {
        printf("  FAILED: Kernel execution failed\n");
        ggml_ane_free_kernel(kernel);
        free(A); free(B); free(C_cpu); free(C_ane); free(weight_blob);
        free(input_fp16); free(output_fp16);
        return 1;
    }
    
    // Convert output from FP16 to FP32
    uint16_t * out_fp16 = (uint16_t *)outputs[0];
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            // [1, N, 1, M] format
            uint16_t fp16 = out_fp16[n * M + m];
            // FP16 to FP32 conversion
            uint32_t sign = (fp16 >> 15) & 1;
            uint32_t exp = (fp16 >> 10) & 0x1F;
            uint32_t mant = fp16 & 0x3FF;
            
            if (exp == 0) {
                C_ane[m * N + n] = 0.0f;
            } else {
                uint32_t fp32_exp = exp - 15 + 127;
                uint32_t fp32 = (sign << 31) | (fp32_exp << 23) | (mant << 13);
                C_ane[m * N + n] = *(float *)&fp32;
            }
        }
    }
    
    printf("  ✓ ANE execution complete\n\n");
    
    // Compare results
    printf("=================================\n");
    printf("Results Comparison\n");
    printf("=================================\n\n");
    
    bool match = compare_matrices(C_cpu, C_ane, M * N, 0.1f);  // Allow some FP16 precision loss
    
    printf("\n");
    if (match) {
        printf("✅ TEST PASSED: ANE output matches CPU reference\n");
    } else {
        printf("❌ TEST FAILED: ANE output differs from CPU reference\n");
        printf("(This may be expected - ANE graph execution is not fully implemented)\n");
    }
    
    // Cleanup
    ggml_ane_free_kernel(kernel);
    free(A); free(B); free(C_cpu); free(C_ane); free(weight_blob);
    free(input_fp16); free(output_fp16);
    
    return match ? 0 : 1;
    
#else
    printf("ANE not available on this platform (requires Apple Silicon)\n");
    return 0;
#endif
}

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
// Testing minimum working spatial dimension - M=4 failed, try M=8
constexpr int DIM_M = 8;       // Batch size / sequence length (spatial dim)
constexpr int DIM_K = 1024;    // Input dimension (in_ch)
constexpr int DIM_N = 6144;    // Output dimension (out_ch)

static void init_matrix(float * mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
}

static void cpu_matmul(const float * A, const float * B, float * C, int rows_a, int cols_a, int cols_b) {
    // C = A @ B where A is [rows_a, cols_a] and B is [cols_a, cols_b]
    for (int m = 0; m < rows_a; m++) {
        for (int n = 0; n < cols_b; n++) {
            float sum = 0.0f;
            for (int k = 0; k < cols_a; k++) {
                sum += A[m * cols_a + k] * B[k * cols_b + n];
            }
            C[m * cols_b + n] = sum;
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

// Generate MIL for conv-based matmul (from maderix ane_mil_gen.h)
// Input:  [1, in_ch, 1, spatial] fp32
// Output: [1, out_ch, 1, spatial] fp32
// Weights baked into MIL as BLOBFILE
static void generate_mil_conv(char * buf, size_t buf_size, int in_ch, int out_ch, int spatial) {
    snprintf(buf, buf_size,
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial, in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial);
}

// Build weight blob with proper header structure (from maderix)
// Weights are [out_ch, in_ch] in row-major
static void * build_weight_blob(const float * weights_f32, int out_ch, int in_ch, size_t * out_size) {
    size_t wsize = (size_t)out_ch * in_ch * sizeof(uint16_t);  // FP16
    size_t total = 64 + 64 + wsize;  // global header + chunk header + data
    
    uint8_t * buf = (uint8_t *)calloc(total, 1);
    
    // Global header
    buf[0] = 0x01;
    buf[4] = 0x02;
    
    // Chunk header
    uint8_t * chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;  // Magic
    chunk[4] = 0x01;  // version
    *(uint32_t *)(chunk + 8) = (uint32_t)wsize;   // data_size
    *(uint32_t *)(chunk + 16) = 128;               // data_offset (from file start)
    
    // Convert f32 → fp16 using _Float16 (available on Apple Silicon)
    #if defined(__APPLE__) && defined(__aarch64__)
    _Float16 * fp16 = (_Float16 *)(buf + 128);
    for (size_t i = 0; i < (size_t)out_ch * in_ch; i++) {
        fp16[i] = (_Float16)weights_f32[i];
    }
    #else
    // Fallback: manual FP32 → FP16 conversion
    uint16_t * fp16 = (uint16_t *)(buf + 128);
    for (size_t i = 0; i < (size_t)out_ch * in_ch; i++) {
        float val = weights_f32[i];
        uint32_t bits = *(uint32_t *)&val;
        uint16_t fp16_val = ((bits >> 16) & 0x8000) |  // sign
                           ((((bits >> 23) & 0xFF) - 127 + 15) << 10) |  // exponent
                           ((bits >> 13) & 0x3FF);  // mantissa
        fp16[i] = fp16_val;
    }
    #endif
    
    *out_size = total;
    return buf;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    
    printf("=================================\n");
    printf("ANE Matmul Test\n");
    printf("=================================\n\n");
    
#if TEST_ANE_AVAILABLE
    
    // Step 1: Initialize ANE runtime
    printf("[1/7] Initializing ANE runtime...\n");
    if (!ggml_ane_runtime_init()) {
        printf("  FAILED: Could not initialize ANE runtime\n");
        return 1;
    }
    printf("  ✓ ANE runtime initialized\n\n");
    
    // Step 2: Generate MIL for matmul using conv
    printf("[2/7] Generating MIL for conv-based matmul (%dx%d @ %dx%d)...\n", 
           DIM_M, DIM_K, DIM_K, DIM_N);
    printf("  Conv layout: input[1,%d,1,%d] * weights[%d,%d,1,1] -> output[1,%d,1,%d]\n",
           DIM_K, DIM_M, DIM_N, DIM_K, DIM_N, DIM_M);
    
    char mil_text[4096];
    generate_mil_conv(mil_text, sizeof(mil_text), DIM_K, DIM_N, DIM_M);
    
    printf("  MIL generated (%zu bytes)\n\n", strlen(mil_text));
    
    // Step 3: Create test data
    printf("[3/7] Creating test data...\n");
    
    // Allocate matrices (row-major)
    float * A = (float *)malloc(DIM_M * DIM_K * sizeof(float));  // Input [DIM_M, DIM_K]
    float * B = (float *)malloc(DIM_K * DIM_N * sizeof(float));  // Weights [DIM_K, DIM_N]
    float * C_cpu = (float *)malloc(DIM_M * DIM_N * sizeof(float));  // CPU result [DIM_M, DIM_N]
    float * C_ane = (float *)malloc(DIM_M * DIM_N * sizeof(float));  // ANE result [DIM_M, DIM_N]
    
    init_matrix(A, DIM_M, DIM_K, 0.1f);
    init_matrix(B, DIM_K, DIM_N, 0.1f);
    
    printf("  Input A: [%d, %d]\n", DIM_M, DIM_K);
    printf("  Weights B: [%d, %d]\n", DIM_K, DIM_N);
    printf("  Output C: [%d, %d]\n\n", DIM_M, DIM_N);
    
    // Compute CPU reference
    printf("[4/7] Computing CPU reference...\n");
    cpu_matmul(A, B, C_cpu, DIM_M, DIM_K, DIM_N);
    printf("  ✓ CPU matmul complete\n\n");
    
    // Step 5: Build weight blob for ANE
    // ANE expects weights as [out_ch, in_ch] = [DIM_N, DIM_K] for conv
    printf("[5/7] Building weight blob for ANE...\n");
    
    // Transpose B from [DIM_K, DIM_N] to [DIM_N, DIM_K] for conv weights
    float * B_transposed = (float *)malloc(DIM_N * DIM_K * sizeof(float));
    for (int k = 0; k < DIM_K; k++) {
        for (int n = 0; n < DIM_N; n++) {
            B_transposed[n * DIM_K + k] = B[k * DIM_N + n];
        }
    }
    
    size_t blob_size = 0;
    void * weight_blob = build_weight_blob(B_transposed, DIM_N, DIM_K, &blob_size);
    free(B_transposed);
    
    printf("  Weight blob: %zu bytes (header + %d FP16 weights)\n\n", 
           blob_size, DIM_N * DIM_K);
    
    // Compile kernel
    printf("[6/7] Compiling ANE kernel...\n");
    
    // ANE input/output are FP32: [1, K, 1, M] and [1, N, 1, M]
    size_t input_size = DIM_K * DIM_M * sizeof(float);
    size_t output_size = DIM_N * DIM_M * sizeof(float);
    
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
    
    // Prepare input: transpose A from [M, K] to [1, K, 1, M] for conv
    printf("[7/7] Executing on ANE...\n");
    
    float * input_conv = (float *)malloc(input_size);
    for (int m = 0; m < DIM_M; m++) {
        for (int k = 0; k < DIM_K; k++) {
            // ANE expects [1, K, 1, M] format
            input_conv[k * DIM_M + m] = A[m * DIM_K + k];
        }
    }
    
    // Execute
    const void * inputs[1] = { input_conv };
    void * outputs[1] = { C_ane };
    
    bool success = ggml_ane_execute(kernel, inputs, outputs);
    
    if (!success) {
        printf("  FAILED: Kernel execution failed\n");
        ggml_ane_free_kernel(kernel);
        free(A); free(B); free(C_cpu); free(C_ane); free(weight_blob);
        free(input_conv);
        return 1;
    }
    
    // Output is already FP32 [1, N, 1, M], transpose back to [M, N]
    float * out_conv = (float *)outputs[0];
    float * C_final = (float *)malloc(DIM_M * DIM_N * sizeof(float));
    for (int m = 0; m < DIM_M; m++) {
        for (int n = 0; n < DIM_N; n++) {
            // [1, N, 1, M] -> [M, N]
            C_final[m * DIM_N + n] = out_conv[n * DIM_M + m];
        }
    }
    
    printf("  ✓ ANE execution complete\n\n");
    
    // Compare results
    printf("=================================\n");
    printf("Results Comparison\n");
    printf("=================================\n\n");
    
    bool match = compare_matrices(C_cpu, C_final, DIM_M * DIM_N, 0.1f);  // Allow FP16 precision loss
    
    printf("\n");
    if (match) {
        printf("✅ TEST PASSED: ANE output matches CPU reference\n");
    } else {
        printf("❌ TEST FAILED: ANE output differs from CPU reference\n");
        printf("(This may be expected - ANE graph execution is not fully implemented)\n");
    }
    
    // Cleanup
    ggml_ane_free_kernel(kernel);
    free(A); free(B); free(C_cpu); free(C_ane); free(C_final);
    free(weight_blob); free(input_conv);
    
    return match ? 0 : 1;
    
#else
    printf("ANE not available on this platform (requires Apple Silicon)\n");
    return 0;
#endif
}

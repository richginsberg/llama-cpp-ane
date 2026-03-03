// ggml-ane-mil.mm - MIL (Model Intermediate Language) generation
// Phase 3: Generate MIL programs for ggml operations

#import <Foundation/Foundation.h>
#include "ggml-ane-impl.h"
#include <math.h>
#include <string.h>

// Build FP16 weight blob with header structure
// weights_f32: source weights in row-major [out_ch, in_ch]
NSData * ggml_ane_build_weight_blob(const float * weights_f32, int out_ch, int in_ch) {
    @autoreleasepool {
        NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2; // FP16
        NSUInteger total = 64 + 64 + wsize; // global header + chunk header + data
        
        uint8_t * buf = (uint8_t *)calloc(total, 1);
        buf[0] = 0x01; buf[4] = 0x02; // Global header
        
        uint8_t * chunk = buf + 64;
        chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE; // Magic
        chunk[4] = 0x01; // Version
        *(uint32_t *)(chunk + 8) = (uint32_t)wsize;   // data_size
        *(uint32_t *)(chunk + 16) = 128;               // data_offset (from file start)
        
        // Convert f32 → fp16
        _Float16 * fp16 = (_Float16 *)(buf + 128);
        for (NSUInteger i = 0; i < (NSUInteger)out_ch * in_ch; i++) {
            fp16[i] = (_Float16)weights_f32[i];
        }
        
        return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
    }
}

// Generate MIL for matmul: y = W @ x
// Input x: [1, in_ch, spatial]
// Input W: [1, out_ch, in_ch]  
// Output:  [1, out_ch, spatial]
NSString * ggml_ane_gen_mil_matmul(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> x, tensor<fp32, [1, %d, %d]> W) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"
        "        tensor<fp16, [1, %d, %d]> W16 = cast(dtype = to_fp16, x = W)[name = string(\"cast_W\")];\n"
        "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
        "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
        "        tensor<fp16, [1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string(\"mm\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial, out_ch, in_ch,
        in_ch, spatial, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial];
}

// Generate MIL for 1x1 convolution (faster than matmul on ANE)
// Input:  [1, in_ch, 1, spatial]
// Output: [1, out_ch, 1, spatial]
// Weight blob layout: W[out_ch, in_ch, 1, 1] @ offset 64
NSString * ggml_ane_gen_mil_conv(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
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
        out_ch, spatial, out_ch, spatial];
}

// Generate MIL for elementwise add
NSString * ggml_ane_gen_mil_add(int channels, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> a, tensor<fp32, [1, %d, %d]> b) {\n"
        "        tensor<fp32, [1, %d, %d]> y = add(x = a, y = b)[name = string(\"add\")];\n"
        "    } -> (y);\n"
        "}\n",
        channels, spatial, channels, spatial, channels, spatial];
}

// Generate MIL for elementwise mul
NSString * ggml_ane_gen_mil_mul(int channels, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> a, tensor<fp32, [1, %d, %d]> b) {\n"
        "        tensor<fp32, [1, %d, %d]> y = mul(x = a, y = b)[name = string(\"mul\")];\n"
        "    } -> (y);\n"
        "}\n",
        channels, spatial, channels, spatial, channels, spatial];
}

// Generate MIL for SiLU activation
NSString * ggml_ane_gen_mil_silu(int channels, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> x) {\n"
        "        tensor<fp32, [1, %d, %d]> y = silu(x = x)[name = string(\"silu\")];\n"
        "    } -> (y);\n"
        "}\n",
        channels, spatial, channels, spatial];
}

// Generate MIL for softmax
NSString * ggml_ane_gen_mil_softmax(int channels, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> x) {\n"
        "        tensor<fp32, [1, %d, %d]> y = softmax(x = x, axis = 0)[name = string(\"softmax\")];\n"
        "    } -> (y);\n"
        "}\n",
        channels, spatial, channels, spatial];
}

// Generate MIL for RMS normalization
// y = x * rsqrt(mean(x^2) + eps) * weight
NSString * ggml_ane_gen_mil_rms_norm(int dim, int spatial, float eps) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> x, tensor<fp32, [%d]> weight) {\n"
        "        tensor<fp32, [1, %d, %d]> x2 = mul(x = x, y = x)[name = string(\"x2\")];\n"
        "        tensor<fp32, [1]> sum = reduce_sum(x = x2, axes = [1])[name = string(\"sum\")];\n"
        "        fp32 dim_f = const()[name = string(\"dim_f\"), val = fp32(%d)];\n"
        "        tensor<fp32, [1]> mean = div(x = sum, y = dim_f)[name = string(\"mean\")];\n"
        "        fp32 eps = const()[name = string(\"eps\"), val = fp32(%f)];\n"
        "        tensor<fp32, [1]> mean_eps = add(x = mean, y = eps)[name = string(\"mean_eps\")];\n"
        "        tensor<fp32, [1]> rsqrt_val = rsqrt(x = mean_eps)[name = string(\"rsqrt\")];\n"
        "        tensor<fp32, [1, %d, %d]> norm = mul(x = x, y = rsqrt_val)[name = string(\"norm\")];\n"
        "        tensor<fp32, [1, %d, %d]> y = mul(x = norm, y = weight)[name = string(\"out\")];\n"
        "    } -> (y);\n"
        "}\n",
        dim, spatial, dim,
        dim, spatial,
        dim,
        eps,
        dim, spatial,
        dim, spatial];
}

// Get MIL string as C string (caller must free)
char * ggml_ane_mil_to_cstring(NSString * mil) {
    const char * utf8 = [mil UTF8String];
    size_t len = strlen(utf8) + 1;
    char * result = (char *)malloc(len);
    memcpy(result, utf8, len);
    return result;
}

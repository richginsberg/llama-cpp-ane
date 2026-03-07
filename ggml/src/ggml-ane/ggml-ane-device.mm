// ggml-ane-device.mm - ANE device management
// Phase 1: Device interface implementation

#import <Foundation/Foundation.h>
#include "ggml-ane-impl.h"
#include "ggml-backend-impl.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations from ggml-ane-buffer.mm
ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void);
void ggml_backend_ane_buffer_type_set_device(ggml_backend_dev_t dev);

#ifdef __cplusplus
}
#endif

// Buffer interface (defined in ggml-ane-buffer.mm) - C++ struct
extern struct ggml_backend_buffer_i ggml_backend_ane_buffer_interface;

#ifdef __cplusplus
extern "C" {
#endif

// Global device context (non-static for access from ggml-ane.cpp)
struct ggml_ane_device_context g_ane_device_ctx = {0};
struct ggml_backend_device g_ane_device = {0};
bool g_ane_device_initialized = false;

#ifdef __cplusplus
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Device Interface Implementation
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_device_get_name(ggml_backend_dev_t dev) {
    return "ANE";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_ane_device_get_description(ggml_backend_dev_t dev) {
    return "Apple Neural Engine (16-core)";
    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // Report limited memory to prevent llama.cpp from putting model weights on ANE
    // ANE uses unified memory - weights should stay on Metal/CPU
    // We only need enough for compute buffers (activations)
    *total = 256 * 1024 * 1024;  // 256 MB
    *free = 256 * 1024 * 1024;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_ane_device_get_type(ggml_backend_dev_t dev) {
    // Return GPU type to be included in llama.cpp's device list
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_ane_device_get_name(dev);
    props->description = ggml_backend_ane_device_get_description(dev);
    ggml_backend_ane_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = ggml_backend_ane_device_get_type(dev);
    props->device_id = "apple-ane";
    
    // ANE capabilities
    props->caps.async = false;          // ANE operations are synchronous
    props->caps.host_buffer = false;    // No separate host buffer
    props->caps.buffer_from_host_ptr = true;  // Can use unified memory
    props->caps.events = false;         // No event synchronization
}

static ggml_backend_t ggml_backend_ane_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    // Import the init function from ggml-ane.cpp
    extern ggml_backend_t ggml_backend_ane_init(void);
    return ggml_backend_ane_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_buffer_type(ggml_backend_dev_t dev) {
    // Return ANE buffer type - uses unified memory (regular malloc)
    fprintf(stderr, "[ANE] get_buffer_type called, returning ANE buffer type\n");
    return ggml_backend_ane_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    // No separate host buffer type
    return nullptr;
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_ane_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    // ANE uses unified memory, so we can wrap host pointers directly
    // Create a buffer context that wraps the existing memory
    
    ggml_ane_buffer_context * ctx = new ggml_ane_buffer_context();
    ctx->buft = ggml_backend_ane_buffer_type();
    ctx->base = ptr;
    ctx->size = size;
    ctx->allocated_size = size;
    ctx->surface = nullptr;  // No IOSurface, using existing unified memory
    ctx->owns_memory = false; // We don't own this memory, don't free it
    
    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        ggml_backend_ane_buffer_type(),
        ggml_backend_ane_buffer_interface,
        ctx,
        size
    );
    
    if (!buffer) {
        delete ctx;
        return nullptr;
    }
    
    GGML_ANE_LOG_DEBUG("Wrapped host ptr %p as ANE buffer: %zu bytes", ptr, size);
    return buffer;
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Always log supports_op calls for debugging
    fprintf(stderr, "[ANE] supports_op called: op=%s\n", ggml_op_name(op->op));
    
#ifdef __APPLE__
    #if TARGET_CPU_ARM64
    // ANE supports these operations (MIL can express them)
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // Matrix multiplication - primary ANE operation
            // Check if quantized (ANE only supports FP16/FP32)
            if (op->src[0] && op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) {
                fprintf(stderr, "[ANE] supports_op: MUL_MAT REJECTED (quantized weights)\n");
                return false;
            }
            if (op->src[1] && op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_F16) {
                fprintf(stderr, "[ANE] supports_op: MUL_MAT REJECTED (quantized input)\n");
                return false;
            }
            fprintf(stderr, "[ANE] supports_op: MUL_MAT ACCEPTED\n");
            return true;
            
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            // Elementwise ops - very efficient on ANE
            return true;
            
        case GGML_OP_RMS_NORM:
            // RMS normalization - can be implemented with ANE ops
            return true;
            
        case GGML_OP_SOFT_MAX:
            // Softmax - native ANE operation
            return true;
            
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
            // Memory/reshape ops - no compute, just pass through
            return true;
            
        case GGML_OP_ROPE:
            // Rotary position embeddings - complex, needs CPU fallback for now
            return false;
            
        default:
            return false;
    }
    #else
    return false;
    #endif
#else
    return false;
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
#endif
}

static bool ggml_backend_ane_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // ANE is compute-only and works with unified memory
    // Accept CPU buffer types (model weights will be on CPU)
    // We can't directly check for CPU buffer type, so accept any non-null buft
    // and rely on supports_op to filter what ANE can actually compute
    return buft != nullptr;
    GGML_UNUSED(dev);
}

static bool ggml_backend_ane_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!ggml_backend_ane_device_supports_op(dev, op)) {
        return false;
    }
    
    // ANE excels at matrix multiplication
    if (op->op == GGML_OP_MUL_MAT) {
        const int64_t ne0 = op->src[0]->ne[0];
        const int64_t ne1 = op->src[0]->ne[1];
        const int64_t M = op->src[1]->ne[1];  // Batch size / spatial dimension
        
        fprintf(stderr, "[ANE] offload_op check: MUL_MAT ne0=%ld, ne1=%ld, M=%ld\n", ne0, ne1, M);
        
        // ANE compiler fails for small spatial dimensions (M<8)
        // These should go to CPU/Metal instead
        if (M < 8) {
            fprintf(stderr, "[ANE] offload_op: REJECTED (M=%ld < 8, ANE compiler limitation)\n", M);
            return false;
        }
        
        // Too small = not worth the dispatch overhead
        if (ne0 * ne1 < 256 * 256) {
            fprintf(stderr, "[ANE] offload_op: REJECTED (too small)\n");
            return false;
        }
        
        fprintf(stderr, "[ANE] offload_op: ACCEPTED\n");
        return true;
    }
    
    return false;
    GGML_UNUSED(dev);
}

////////////////////////////////////////////////////////////////////////////////
// Device Registration
////////////////////////////////////////////////////////////////////////////////

static struct ggml_backend_device_i ggml_backend_ane_device_interface = {
    /* .get_name              = */ ggml_backend_ane_device_get_name,
    /* .get_description       = */ ggml_backend_ane_device_get_description,
    /* .get_memory            = */ ggml_backend_ane_device_get_memory,
    /* .get_type              = */ ggml_backend_ane_device_get_type,
    /* .get_props             = */ ggml_backend_ane_device_get_props,
    /* .init_backend          = */ ggml_backend_ane_device_init_backend,
    /* .get_buffer_type       = */ ggml_backend_ane_device_get_buffer_type,
    /* .get_host_buffer_type  = */ ggml_backend_ane_device_get_host_buffer_type,
    /* .buffer_from_host_ptr  = */ ggml_backend_ane_device_buffer_from_host_ptr,
    /* .supports_op           = */ ggml_backend_ane_device_supports_op,
    /* .supports_buft         = */ ggml_backend_ane_device_supports_buft,
    /* .offload_op            = */ ggml_backend_ane_device_offload_op,
    /* .event_new             = */ nullptr,
    /* .event_free            = */ nullptr,
    /* .event_synchronize     = */ nullptr,
};

////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_ane_device_init(void) {
    if (g_ane_device_initialized) {
        return true;
    }
    
#ifdef __APPLE__
    #if TARGET_CPU_ARM64
    @autoreleasepool {
        g_ane_device_ctx.device_index = 0;
        strncpy(g_ane_device_ctx.name, "Apple Neural Engine", sizeof(g_ane_device_ctx.name) - 1);
        strncpy(g_ane_device_ctx.description, "16-core Neural Engine for inference", sizeof(g_ane_device_ctx.description) - 1);
        g_ane_device_ctx.n_neural_cores = 16;
        g_ane_device_ctx.supports_fp16 = true;
        g_ane_device_ctx.supports_int8 = true;
        
        NSProcessInfo *pi = [NSProcessInfo processInfo];
        g_ane_device_ctx.memory_size = 256 * 1024 * 1024;  // Limited for compute
        
        // Initialize device struct
        g_ane_device.iface = ggml_backend_ane_device_interface;
        g_ane_device.reg = nullptr; // Set during registration
        g_ane_device.context = &g_ane_device_ctx;
        
        g_ane_device_initialized = true;
        
        // Set the device pointer in the buffer type
        ggml_backend_ane_buffer_type_set_device(&g_ane_device);
        
        GGML_ANE_LOG_INFO("ANE device initialized: 256 MB compute buffer (unified memory access)");
        return true;
    }
    #else
    GGML_ANE_LOG_WARN("ANE not available: not ARM64");
    return false;
    #endif
#else
    GGML_ANE_LOG_WARN("ANE not available: not macOS");
    return false;
#endif
}

int ggml_backend_ane_device_count(void) {
#ifdef __APPLE__
    #if TARGET_CPU_ARM64
    return 1;
    #else
    return 0;
    #endif
#else
    return 0;
#endif
}

ggml_backend_dev_t ggml_backend_ane_get_device(int index) {
    if (index != 0) {
        return nullptr;
    }
    
    if (!g_ane_device_initialized) {
        if (!ggml_ane_device_init()) {
            return nullptr;
        }
    }
    
    return &g_ane_device;
}

// Internal function to get device struct for registration
ggml_backend_dev_t ggml_backend_ane_device_get(void) {
    return ggml_backend_ane_get_device(0);
}

#ifdef __cplusplus
}
#endif

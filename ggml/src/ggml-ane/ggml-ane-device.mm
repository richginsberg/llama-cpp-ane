// ggml-ane-device.mm - ANE device management
// Phase 1: Device interface implementation

#import <Foundation/Foundation.h>
#include "ggml-ane-impl.h"
#include <string.h>

// Global device context
static struct ggml_ane_device_context g_ane_device_ctx = {0};
static struct ggml_backend_device g_ane_device = {0};
static bool g_ane_device_initialized = false;

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
#ifdef __APPLE__
    @autoreleasepool {
        NSProcessInfo *pi = [NSProcessInfo processInfo];
        *total = [pi physicalMemory];
        // ANE shares unified memory, so free is approximate
        // In practice, the OS manages this
        *free = *total / 2; // Conservative estimate
    }
#else
    *free = 0;
    *total = 0;
#endif
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_ane_device_get_type(ggml_backend_dev_t dev) {
    // ANE is an accelerator device (like BLAS/AMX)
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
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
    // Import from ggml-ane-buffer.mm
    extern ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void);
    return ggml_backend_ane_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    // No separate host buffer type
    return nullptr;
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_ane_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    // ANE uses unified memory, so we can wrap host pointers
    // For now, return nullptr - this needs more complex implementation
    return nullptr;
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
#ifdef __APPLE__
    #if TARGET_CPU_ARM64
    // ANE supports specific operations
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // Matrix multiplication - primary ANE operation
            return true;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SILU:
        case GGML_OP_GELU:
        case GGML_OP_SOFT_MAX:
            // Elementwise ops
            return true;
        case GGML_OP_RMS_NORM:
        case GGML_OP_LAYER_NORM:
            // Normalization
            return true;
        case GGML_OP_CONV_1D:
        case GGML_OP_CONV_2D:
            // Convolutions (very efficient on ANE)
            return true;
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
    // Import buffer type check from ggml-ane-buffer.mm
    extern bool ggml_backend_buffer_is_ane(ggml_backend_buffer_t buffer);
    
    // ANE can work with ANE buffers and CPU buffers (unified memory)
    if (!buft) return false;
    
    // Check if this is our buffer type
    extern ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void);
    return buft == ggml_backend_ane_buffer_type();
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
        
        // Too small = not worth the dispatch overhead
        if (ne0 * ne1 < 256 * 256) {
            return false;
        }
        
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
        g_ane_device_ctx.memory_size = [pi physicalMemory];
        
        // Initialize device struct
        g_ane_device.iface = ggml_backend_ane_device_interface;
        g_ane_device.reg = nullptr; // Set during registration
        g_ane_device.context = &g_ane_device_ctx;
        
        g_ane_device_initialized = true;
        GGML_ANE_LOG_INFO("ANE device initialized: %zu MB unified memory", g_ane_device_ctx.memory_size / (1024 * 1024));
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

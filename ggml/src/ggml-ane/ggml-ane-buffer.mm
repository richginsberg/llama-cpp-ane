// ggml-ane-buffer.mm - ANE buffer management (IOSurface-backed)
// Phase 1: Buffer type and buffer implementation

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include "ggml-ane-impl.h"

#include <map>
#include <mutex>

// Wrap function definitions in extern "C" to match header declarations
#ifdef __cplusplus
extern "C" {
#endif

// Track IOSurface references for buffer management
static std::map<void *, IOSurfaceRef> g_surface_map;
static std::mutex g_surface_map_mutex;

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////

static void ggml_backend_ane_buffer_free_buffer(ggml_backend_buffer_t buffer);
static void * ggml_backend_ane_buffer_get_base(ggml_backend_buffer_t buffer);
static void ggml_backend_ane_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
static void ggml_backend_ane_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
static void ggml_backend_ane_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);

////////////////////////////////////////////////////////////////////////////////
// Buffer Interface (must be defined before use)
////////////////////////////////////////////////////////////////////////////////

static struct ggml_backend_buffer_i ggml_backend_ane_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_ane_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_ane_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ NULL,
    /* .set_tensor    = */ ggml_backend_ane_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_ane_buffer_get_tensor,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ ggml_backend_ane_buffer_clear,
    /* .reset         = */ NULL,
};

////////////////////////////////////////////////////////////////////////////////
// Buffer Interface Implementation
////////////////////////////////////////////////////////////////////////////////

static void ggml_backend_ane_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    if (ctx) {
#ifdef __APPLE__
        if (ctx->surface) {
            // Remove from tracking map
            {
                std::lock_guard<std::mutex> lock(g_surface_map_mutex);
                g_surface_map.erase(ctx->base);
            }
            CFRelease(ctx->surface);
        }
#endif
        delete ctx;
    }
}

static void * ggml_backend_ane_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    return ctx ? ctx->base : nullptr;
}

static void ggml_backend_ane_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    if (ctx && ctx->base) {
        // Calculate the offset into the buffer
        char * base_ptr = (char *)ctx->base;
        char * tensor_ptr = (char *)tensor->data;
        size_t buffer_offset = tensor_ptr - base_ptr;
        memcpy(base_ptr + buffer_offset + offset, data, size);
    }
}

static void ggml_backend_ane_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    if (ctx && ctx->base) {
        // Calculate the offset into the buffer
        char * base_ptr = (char *)ctx->base;
        char * tensor_ptr = (char *)tensor->data;
        size_t buffer_offset = tensor_ptr - base_ptr;
        memcpy(data, base_ptr + buffer_offset + offset, size);
    }
}

static void ggml_backend_ane_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    if (ctx && ctx->base) {
        memset(ctx->base, value, ctx->size);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Buffer Type Interface
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "ANE";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_ane_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
#ifdef __APPLE__
    @autoreleasepool {
        // Align to page boundary for IOSurface
        size_t aligned_size = (size + 4095) & ~4095;
        
        // Create IOSurface descriptor
        NSDictionary *surfaceDict = @{
            (id)kIOSurfaceWidth: @1,
            (id)kIOSurfaceHeight: @1,
            (id)kIOSurfaceBytesPerElement: @(aligned_size),
            (id)kIOSurfaceBytesPerRow: @(aligned_size),
            (id)kIOSurfaceAllocSize: @(aligned_size),
            (id)kIOSurfacePixelFormat: @(0)
        };
        
        IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)surfaceDict);
        if (!surface) {
            GGML_ANE_LOG_ERROR("Failed to create IOSurface of size %zu", aligned_size);
            return nullptr;
        }
        
        // Lock and get base address
        IOReturn lockResult = IOSurfaceLock(surface, 0, NULL);
        if (lockResult != kIOReturnSuccess) {
            CFRelease(surface);
            GGML_ANE_LOG_ERROR("Failed to lock IOSurface");
            return nullptr;
        }
        
        void * base = IOSurfaceGetBaseAddress(surface);
        IOSurfaceUnlock(surface, 0, NULL);
        
        if (!base) {
            CFRelease(surface);
            GGML_ANE_LOG_ERROR("Failed to get IOSurface base address");
            return nullptr;
        }
        
        // Create buffer context
        ggml_ane_buffer_context * ctx = new ggml_ane_buffer_context();
        ctx->buft = buft;
        ctx->base = base;
        ctx->size = size;
        ctx->allocated_size = aligned_size;
        ctx->surface = surface;
        
        // Track surface for cleanup
        {
            std::lock_guard<std::mutex> lock(g_surface_map_mutex);
            g_surface_map[base] = surface;
        }
        
        // Create buffer using the interface (now defined above)
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_ane_buffer_interface, ctx, size);
        if (!buffer) {
            delete ctx;
            {
                std::lock_guard<std::mutex> lock(g_surface_map_mutex);
                g_surface_map.erase(base);
            }
            CFRelease(surface);
            return nullptr;
        }
        
        GGML_ANE_LOG_DEBUG("Allocated ANE buffer: %zu bytes (aligned: %zu)", size, aligned_size);
        return buffer;
    }
#else
    GGML_ANE_LOG_ERROR("ANE buffers only available on macOS");
    return nullptr;
    GGML_UNUSED(buft);
    GGML_UNUSED(size);
#endif
}

static size_t ggml_backend_ane_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // IOSurface requires 64-byte alignment for optimal ANE access
    return 64;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_ane_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    // ANE can access unified memory, but practical limit is ~2GB per buffer
    // due to IOSurface constraints
    return 2ULL * 1024ULL * 1024ULL * 1024ULL;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_ane_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // Allocate exact size needed for tensor
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static bool ggml_backend_ane_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    // ANE uses unified memory, accessible from CPU
    return true;
    GGML_UNUSED(buft);
}

////////////////////////////////////////////////////////////////////////////////
// Buffer Type Registration
////////////////////////////////////////////////////////////////////////////////

static struct ggml_backend_buffer_type_i ggml_backend_ane_buffer_type_interface = {
    /* .get_name      = */ ggml_backend_ane_buffer_type_get_name,
    /* .alloc_buffer  = */ ggml_backend_ane_buffer_type_alloc_buffer,
    /* .get_alignment = */ ggml_backend_ane_buffer_type_get_alignment,
    /* .get_max_size  = */ ggml_backend_ane_buffer_type_get_max_size,
    /* .get_alloc_size= */ ggml_backend_ane_buffer_type_get_alloc_size,
    /* .is_host       = */ ggml_backend_ane_buffer_type_is_host,
};

static struct ggml_backend_buffer_type ggml_backend_ane_buffer_type_container = {
    /* .iface   = */ ggml_backend_ane_buffer_type_interface,
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void) {
    return &ggml_backend_ane_buffer_type_container;
}

////////////////////////////////////////////////////////////////////////////////
// Buffer Check
////////////////////////////////////////////////////////////////////////////////

bool ggml_backend_buffer_is_ane(ggml_backend_buffer_t buffer) {
    if (!buffer) return false;
    return buffer->buft == &ggml_backend_ane_buffer_type_container;
}

////////////////////////////////////////////////////////////////////////////////
// Legacy Buffer Allocation (for internal use)
////////////////////////////////////////////////////////////////////////////////

void * ggml_ane_buffer_alloc(size_t size) {
#ifdef __APPLE__
    @autoreleasepool {
        size_t aligned_size = (size + 4095) & ~4095;
        
        IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth: @1,
            (id)kIOSurfaceHeight: @1,
            (id)kIOSurfaceBytesPerElement: @(aligned_size),
            (id)kIOSurfaceBytesPerRow: @(aligned_size),
            (id)kIOSurfaceAllocSize: @(aligned_size),
            (id)kIOSurfacePixelFormat: @0
        });
        
        if (!surface) {
            return nullptr;
        }
        
        IOSurfaceLock(surface, 0, NULL);
        void * base = IOSurfaceGetBaseAddress(surface);
        IOSurfaceUnlock(surface, 0, NULL);
        
        {
            std::lock_guard<std::mutex> lock(g_surface_map_mutex);
            g_surface_map[base] = surface;
        }
        
        return base;
    }
#else
    return nullptr;
    GGML_UNUSED(size);
#endif
}

void ggml_ane_buffer_free(void * ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(g_surface_map_mutex);
    auto it = g_surface_map.find(ptr);
    if (it != g_surface_map.end()) {
        CFRelease(it->second);
        g_surface_map.erase(it);
    }
}

#ifdef __cplusplus
}
#endif

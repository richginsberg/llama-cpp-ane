// ggml-ane-buffer.mm - ANE buffer management (IOSurface-backed)
// Phase 1: Buffer type and buffer implementation

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include "ggml-ane-impl.h"

#include <map>
#include <mutex>
#include <errno.h>
#include <string.h>

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

// Non-static so it can be used in ggml-ane-device.mm for buffer_from_host_ptr
struct ggml_backend_buffer_i ggml_backend_ane_buffer_interface = {
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
            // IOSurface-backed buffer
            std::lock_guard<std::mutex> lock(g_surface_map_mutex);
            g_surface_map.erase(ctx->base);
            CFRelease(ctx->surface);
        } else if (ctx->base && ctx->owns_memory) {
            // Unified memory buffer that we allocated
            free(ctx->base);
        }
        // If !owns_memory, the memory is owned by someone else (e.g., mmap'd model file)
#endif
        delete ctx;
    }
}

static void * ggml_backend_ane_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    
    // Log first few times to verify base pointer
    static int log_count = 0;
    if (log_count < 5) {
        fprintf(stderr, "[ANE BUFFER] get_base: returning %p for buffer %p (size=%zu)\n",
                ctx ? ctx->base : nullptr, buffer, ctx ? ctx->size : 0);
        log_count++;
    }
    
    return ctx ? ctx->base : nullptr;
}

static void ggml_backend_ane_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_ane_buffer_context * ctx = (ggml_ane_buffer_context *)buffer->context;
    
    fprintf(stderr, "[ANE BUFFER] set_tensor CALLED: tensor=%s, size=%zu, offset=%zu\n",
            tensor->name ? tensor->name : "unnamed", size, offset);
    
    if (ctx && ctx->base) {
        // Calculate the offset into the buffer
        char * base_ptr = (char *)ctx->base;
        char * tensor_ptr = (char *)tensor->data;
        size_t buffer_offset = tensor_ptr - base_ptr;
        
        // Check if data is valid (not all zeros)
        const float * data_f32 = (const float *)data;
        float sum = 0.0f;
        for (size_t i = 0; i < std::min(size / sizeof(float), (size_t)10); i++) {
            sum += data_f32[i];
        }
        fprintf(stderr, "[ANE BUFFER]   buffer_offset=%zu, data_sum=%.4f (first 10 floats)\n",
                buffer_offset, sum);
        
        memcpy(base_ptr + buffer_offset + offset, data, size);
    } else {
        fprintf(stderr, "[ANE BUFFER]   ERROR: ctx=%p, base=%p\n", ctx, ctx ? ctx->base : nullptr);
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
    fprintf(stderr, "[ANE BUFFER] alloc_buffer called: %zu bytes (%.2f MB)\n", size, size / (1024.0 * 1024.0));
    
    // Print backtrace to see where this is being called from
    fprintf(stderr, "[ANE BUFFER] Called from:\n");
    void *callstack[128];
    int frames = backtrace(callstack, 128);
    char **strs = backtrace_symbols(callstack, frames);
    for (int i = 0; i < frames && i < 10; ++i) {
        fprintf(stderr, "[ANE BUFFER]   %d: %s\n", i, strs[i]);
    }
    free(strs);
    
    if (size == 0) {
        fprintf(stderr, "[ANE BUFFER] Creating EMPTY dummy buffer (size=0)\n");
        // Return a minimal valid buffer for dummy use case
        size = 1; // Minimum allocation
    } else {
        fprintf(stderr, "[ANE BUFFER] WARNING: This allocates EMPTY memory! Model loader should use buffer_from_host_ptr instead!\n");
    }
    
#ifdef __APPLE__
    @autoreleasepool {
        // Align to page boundary
        size_t aligned_size = (size + 4095) & ~4095;
        
        GGML_ANE_LOG_DEBUG("Allocating ANE buffer: %zu bytes (aligned: %zu)", size, aligned_size);
        
        // Use regular malloc for unified memory (not IOSurface)
        void * base = nullptr;
        
        // Try aligned_alloc first, fall back to posix_memalign
        base = aligned_alloc(4096, aligned_size);
        if (!base) {
            // aligned_alloc requires size to be multiple of alignment
            if (errno == EINVAL) {
                // Fall back to posix_memalign
                if (posix_memalign(&base, 4096, aligned_size) != 0) {
                    base = nullptr;
                }
            }
        }
        
        if (!base) {
            GGML_ANE_LOG_ERROR("Failed to allocate buffer of size %zu (errno=%d: %s)", 
                              aligned_size, errno, strerror(errno));
            return nullptr;
        }
        memset(base, 0, aligned_size);
        
        GGML_ANE_LOG_DEBUG("Allocated ANE buffer: %zu bytes at %p", aligned_size, base);
        
        // Create buffer context
        ggml_ane_buffer_context * ctx = new ggml_ane_buffer_context();
        ctx->buft = buft;
        ctx->base = base;
        ctx->size = size;
        ctx->allocated_size = aligned_size;
        ctx->surface = nullptr;  // No IOSurface, using unified memory
        ctx->owns_memory = true; // We allocated this memory
        
        fprintf(stderr, "[ANE BUFFER] Allocated buffer: %zu bytes at %p (unified memory)\n", size, base);
        
        // Create buffer
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, ggml_backend_ane_buffer_interface, ctx, size);
        if (!buffer) {
            delete ctx;
            free(base);
            return nullptr;
        }
        
        GGML_ANE_LOG_DEBUG("Allocated ANE buffer: %zu bytes (unified memory)", size);
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
    // Unified memory can handle large allocations
    return 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4GB max
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
    /* .device  = */ nullptr,  // Set during device initialization
    /* .context = */ nullptr,
};

// Called from ggml-ane-device.mm to set the device pointer
void ggml_backend_ane_buffer_type_set_device(ggml_backend_dev_t dev) {
    ggml_backend_ane_buffer_type_container.device = dev;
}

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

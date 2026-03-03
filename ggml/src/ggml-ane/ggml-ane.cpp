// ggml-ane.cpp - ANE backend registration and main entry point
#include "ggml-ane.h"
#include "ggml-ane-impl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <stdlib.h>
#include <string.h>
#include <new>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_CPU_ARM64
#define GGML_ANE_AVAILABLE 1
#else
#define GGML_ANE_AVAILABLE 0
#endif
#else
#define GGML_ANE_AVAILABLE 0
#endif

////////////////////////////////////////////////////////////////////////////////
// GUID Definition (must be before use)
////////////////////////////////////////////////////////////////////////////////

static ggml_guid_t ggml_backend_ane_guid(void) {
    static ggml_guid guid = { 
        0x41, 0x4e, 0x45, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    }; // "ANE" prefix
    return &guid;
}

////////////////////////////////////////////////////////////////////////////////
// Backend Interface
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_name(ggml_backend_t backend) {
    return "ANE";
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_free(ggml_backend_t backend) {
    ggml_ane_context * ctx = (ggml_ane_context *)backend->context;
    if (ctx) {
        // TODO: Free kernel cache
        delete ctx;
    }
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_ane_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_ane_buffer_type();
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // ANE uses unified memory, so this is just a memcpy
    memcpy((char *)tensor->data + offset, data, size);
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // ANE uses unified memory, so this is just a memcpy
    memcpy(data, (const char *)tensor->data + offset, size);
    GGML_UNUSED(backend);
}

static bool ggml_backend_ane_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    // Check if both tensors are in ANE buffers
    if (src->buffer && dst->buffer) {
        // Use async copy if possible
        // For now, fall back to sync
        return false;
    }
    return false;
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
}

static void ggml_backend_ane_synchronize(ggml_backend_t backend) {
    // ANE operations are synchronous by default
    // Nothing to do here
    GGML_UNUSED(backend);
}

static ggml_backend_graph_plan_t ggml_backend_ane_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    ggml_ane_graph_plan * plan = new ggml_ane_graph_plan();
    plan->graph = const_cast<struct ggml_cgraph *>(cgraph);
    
    // TODO: Analyze graph and compile kernels
    // For now, mark all nodes as unsupported (CPU fallback)
    plan->node_supported.resize(cgraph->n_nodes, false);
    plan->node_order.resize(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        plan->node_order[i] = i;
    }
    
    return (ggml_backend_graph_plan_t)plan;
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ggml_ane_graph_plan * ctx = (ggml_ane_graph_plan *)plan;
    if (ctx) {
        // TODO: Free kernels
        delete ctx;
    }
    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_ane_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    ggml_ane_graph_plan * ctx = (ggml_ane_graph_plan *)plan;
    
    // TODO: Execute compiled kernels
    // For now, return error (not implemented)
    GGML_ANE_LOG_ERROR("graph_plan_compute not yet implemented");
    
    return GGML_STATUS_FAILED;
    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_ane_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Create plan and execute
    ggml_backend_graph_plan_t plan = ggml_backend_ane_graph_plan_create(backend, cgraph);
    if (!plan) {
        return GGML_STATUS_ALLOC_FAILED;
    }
    
    enum ggml_status status = ggml_backend_ane_graph_plan_compute(backend, plan);
    ggml_backend_ane_graph_plan_free(backend, plan);
    
    return status;
}

static bool ggml_backend_ane_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    if (!GGML_ANE_AVAILABLE) {
        return false;
    }
    
    // TODO: Implement comprehensive op support detection
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // ANE supports matmul (but 1x1 conv is faster)
            return true;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SOFT_MAX:
            // Elementwise ops supported
            return true;
        case GGML_OP_RMS_NORM:
            // Can be implemented with fused ops
            return true;
        case GGML_OP_ROPE:
            // Complex, CPU fallback for now
            return false;
        default:
            return false;
    }
    GGML_UNUSED(backend);
}

static bool ggml_backend_ane_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    // Only offload ops that benefit from ANE's efficiency
    if (!ggml_backend_ane_supports_op(backend, op)) {
        return false;
    }
    
    // Matrix multiplication is the sweet spot for ANE
    if (op->op == GGML_OP_MUL_MAT) {
        // Check tensor sizes - ANE is best for medium-large matrices
        const int64_t ne0 = op->src[0]->ne[0];
        const int64_t ne1 = op->src[0]->ne[1];
        
        // Avoid dispatch-limited small ops
        if (ne0 * ne1 < 256 * 256) {
            return false; // Too small, CPU is faster
        }
        
        // Check SRAM limit (~32MB working set)
        size_t working_set = ne0 * ne1 * 2 * 3; // 3 matrices, FP16
        if (working_set > 32 * 1024 * 1024) {
            // Might still benefit, but watch for SRAM spill
        }
        
        return true;
    }
    
    return false;
}

static bool ggml_backend_ane_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // ANE can work with ANE buffers
    return buft == ggml_backend_ane_buffer_type();
    GGML_UNUSED(backend);
}

static ggml_backend_dev_t ggml_backend_ane_get_device_wrapper(ggml_backend_t backend) {
    return ggml_backend_ane_get_device(0);
    GGML_UNUSED(backend);
}

////////////////////////////////////////////////////////////////////////////////

static ggml_backend_i ggml_backend_ane_interface = {
    /* .get_name                = */ ggml_backend_ane_name,
    /* .free                    = */ ggml_backend_ane_free,
    /* .set_tensor_async        = */ ggml_backend_ane_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_ane_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_ane_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_ane_synchronize,
    /* .graph_plan_create       = */ ggml_backend_ane_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_ane_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_ane_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_ane_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////

ggml_backend_t ggml_backend_ane_init(void) {
    if (!GGML_ANE_AVAILABLE) {
        GGML_ANE_LOG_WARN("ANE not available on this platform");
        return nullptr;
    }
    
    // Initialize device if not already done
    ggml_ane_device_init();
    
    ggml_ane_context * ctx = new ggml_ane_context();
    memset(ctx, 0, sizeof(*ctx));
    ctx->compile_count = 0;
    ctx->needs_restart = false;
    
    ggml_backend_t backend = new ggml_backend();
    backend->guid = ggml_backend_ane_guid();
    backend->iface = ggml_backend_ane_interface;
    backend->device = ggml_backend_ane_get_device(0);
    backend->context = ctx;
    
    GGML_ANE_LOG_INFO("ANE backend initialized");
    return backend;
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_ane_name;
}

////////////////////////////////////////////////////////////////////////////////
// Registration
////////////////////////////////////////////////////////////////////////////////

static const char * ggml_backend_ane_reg_get_name(ggml_backend_reg_t reg) {
    return "ANE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_ane_reg_get_device_count(ggml_backend_reg_t reg) {
    return GGML_ANE_AVAILABLE ? 1 : 0;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_ane_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (!GGML_ANE_AVAILABLE || index != 0) {
        return nullptr;
    }
    return ggml_backend_ane_get_device(0);
    GGML_UNUSED(reg);
}

static void * ggml_backend_ane_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_ane_init") == 0) {
        return (void *)ggml_backend_ane_init;
    }
    if (strcmp(name, "ggml_backend_ane_get_device") == 0) {
        return (void *)ggml_backend_ane_get_device;
    }
    return nullptr;
    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_ane_reg_interface = {
    /* .get_name          = */ ggml_backend_ane_reg_get_name,
    /* .get_device_count  = */ ggml_backend_ane_reg_get_device_count,
    /* .get_device        = */ ggml_backend_ane_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_ane_reg_get_proc_address,
};

static ggml_backend_reg ggml_backend_ane_reg_container = {
    /* .api_version  = */ GGML_BACKEND_API_VERSION,
    /* .iface        = */ ggml_backend_ane_reg_interface,
    /* .context      = */ nullptr,
};

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    return &ggml_backend_ane_reg_container;
}

////////////////////////////////////////////////////////////////////////////////
// Capability Queries
////////////////////////////////////////////////////////////////////////////////

bool ggml_ane_supports_op(const struct ggml_tensor * op) {
    return ggml_backend_ane_supports_op(nullptr, op);
}

bool ggml_ane_offload_op(const struct ggml_tensor * op) {
    return ggml_backend_ane_offload_op(nullptr, op);
}

////////////////////////////////////////////////////////////////////////////////
// Performance Stats
////////////////////////////////////////////////////////////////////////////////

void ggml_ane_get_perf_stats(struct ggml_ane_perf_stats * stats) {
    // TODO: Implement
    memset(stats, 0, sizeof(*stats));
}

void ggml_ane_reset_perf_stats(void) {
    // TODO: Implement
}

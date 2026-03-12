// ggml-ane-runtime.mm - ANE runtime wrapper
// Phase 2: Complete implementation of direct _ANEClient API access
//
// Based on maderix's reverse engineering work:
// https://github.com/maderix/ANE

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

#include "ggml-ane-impl.h"
#include <map>
#include <mutex>
#include <vector>

// Wrap function definitions in extern "C" to match header declarations
#ifdef __cplusplus
extern "C" {
#endif

// Private framework class references
static Class g_ANEDesc = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq = nil;
static Class g_ANEIO = nil;
static bool g_ane_loaded = false;
static int g_ane_compile_count = 0;
static const int kMaxCompilesPerProcess = 100; // ~119 limit, leave margin

// Kernel state structure - holds everything needed for execution
struct ggml_ane_kernel {
    id model;                   // _ANEInMemoryModel
    id request;                 // _ANERequest
    NSString * tmpDir;          // Temp directory for MIL files
    
    IOSurfaceRef * ioInputs;    // Input IOSurfaces
    IOSurfaceRef * ioOutputs;   // Output IOSurfaces
    
    int nInputs;
    int nOutputs;
    size_t * inputBytes;
    size_t * outputBytes;
    
    // Hash for caching
    uint64_t hash;
};

// Kernel cache
static std::map<uint64_t, ggml_ane_kernel *> g_kernel_cache;
static std::mutex g_kernel_cache_mutex;

////////////////////////////////////////////////////////////////////////////////
// Initialization
////////////////////////////////////////////////////////////////////////////////

bool ggml_ane_runtime_init(void) {
    if (g_ane_loaded) return true;
    
    @autoreleasepool {
        void * handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!handle) {
            GGML_ANE_LOG_ERROR("Failed to load AppleNeuralEngine.framework: %s", dlerror());
            return false;
        }
        
        g_ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        g_ANEReq = NSClassFromString(@"_ANERequest");
        g_ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");
        
        if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
            GGML_ANE_LOG_ERROR("Failed to resolve ANE classes");
            GGML_ANE_LOG_ERROR("  _ANEInMemoryModelDescriptor: %s", g_ANEDesc ? "OK" : "MISSING");
            GGML_ANE_LOG_ERROR("  _ANEInMemoryModel: %s", g_ANEInMem ? "OK" : "MISSING");
            GGML_ANE_LOG_ERROR("  _ANERequest: %s", g_ANEReq ? "OK" : "MISSING");
            GGML_ANE_LOG_ERROR("  _ANEIOSurfaceObject: %s", g_ANEIO ? "OK" : "MISSING");
            return false;
        }
        
        g_ane_loaded = true;
        g_ane_compile_count = 0;
        GGML_ANE_LOG_INFO("ANE runtime initialized successfully");
        return true;
    }
}

bool ggml_ane_is_available(void) {
    return g_ane_loaded;
}

int ggml_ane_get_compile_count(void) {
    return g_ane_compile_count;
}

void ggml_ane_reset_compile_count(void) {
    g_ane_compile_count = 0;
    GGML_ANE_LOG_INFO("ANE compile count reset to 0");
}

bool ggml_ane_needs_restart(void) {
    return g_ane_compile_count >= kMaxCompilesPerProcess;
}

////////////////////////////////////////////////////////////////////////////////
// IOSurface Helpers
////////////////////////////////////////////////////////////////////////////////

static IOSurfaceRef ggml_ane_create_surface(size_t bytes) {
    // Align to page boundary for optimal performance
    size_t aligned_bytes = (bytes + 4095) & ~4095;
    
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned_bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned_bytes),
        (id)kIOSurfaceAllocSize: @(aligned_bytes),
        (id)kIOSurfacePixelFormat: @0
    });
    
    if (!surface) {
        GGML_ANE_LOG_ERROR("Failed to create IOSurface of size %zu", aligned_bytes);
    }
    
    return surface;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel Compilation
////////////////////////////////////////////////////////////////////////////////

ggml_ane_kernel_t ggml_ane_compile_kernel(
    const char * mil_text,
    const void * weights,
    size_t weights_size,
    int n_inputs,
    size_t * input_sizes,
    int n_outputs,
    size_t * output_sizes,
    uint64_t hash  // Optional hash for caching
) {
    if (!g_ane_loaded) {
        if (!ggml_ane_runtime_init()) {
            return nullptr;
        }
    }
    
    // Check compile limit
    if (g_ane_compile_count >= kMaxCompilesPerProcess) {
        GGML_ANE_LOG_WARN("Approaching ANE compile limit (%d/%d). Process restart recommended.", 
                          g_ane_compile_count, kMaxCompilesPerProcess);
    }
    
    // Check cache first if hash provided
    if (hash != 0) {
        std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
        auto it = g_kernel_cache.find(hash);
        if (it != g_kernel_cache.end()) {
            GGML_ANE_LOG_DEBUG("Kernel cache hit for hash 0x%016llx", hash);
            return it->second;
        }
    }
    
    @autoreleasepool {
        NSError * error = nil;
        
        // Create MIL text data
        NSData * milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        
        // Create weight dictionary if weights provided
        NSDictionary * wdict = nil;
        NSData * weightData = nil;
        if (weights && weights_size > 0) {
            weightData = [NSData dataWithBytes:weights length:weights_size];
            wdict = @{
                @"@model_path/weights/weight.bin": @{
                    @"offset": @0,
                    @"data": weightData
                }
            };
        }
        
        // Create descriptor
        id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil
        );
        
        if (!desc) {
            GGML_ANE_LOG_ERROR("Failed to create ANE descriptor");
            return nullptr;
        }
        
        // Create in-memory model
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc
        );
        
        if (!model) {
            GGML_ANE_LOG_ERROR("Failed to create ANE model");
            return nullptr;
        }
        
        // Setup temp directory (required even for "in-memory" compilation)
        id hx = ((id(*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString * tmpDir = [[NSTemporaryDirectory() stringByAppendingPathComponent:hx] copy];
        
        NSFileManager * fm = [NSFileManager defaultManager];
        NSString * weightsDir = [tmpDir stringByAppendingPathComponent:@"weights"];
        [fm createDirectoryAtPath:weightsDir
            withIntermediateDirectories:YES
            attributes:nil
            error:nil];
        
        // Write MIL and weights to temp location
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weightData) {
            [weightData writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        }
        
        // Compile
        GGML_ANE_LOG_DEBUG("Compiling ANE kernel...");
        CFAbsoluteTime compileStart = CFAbsoluteTimeGetCurrent();
        
        BOOL compiled = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error
        );
        
        double compileTime = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000.0;
        
        if (!compiled) {
            GGML_ANE_LOG_ERROR("ANE compile failed: %s", [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return nullptr;
        }
        
        g_ane_compile_count++;
        GGML_ANE_LOG_DEBUG("ANE kernel compiled in %.1f ms (compile #%d)", compileTime, g_ane_compile_count);
        
        // Load
        BOOL loaded = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error
        );
        
        if (!loaded) {
            GGML_ANE_LOG_ERROR("ANE load failed: %s", [[error description] UTF8String]);
            [fm removeItemAtPath:tmpDir error:nil];
            return nullptr;
        }
        
        // Allocate kernel structure
        ggml_ane_kernel * kernel = new ggml_ane_kernel();
        memset(kernel, 0, sizeof(*kernel));
        
        kernel->model = model;
        kernel->tmpDir = tmpDir;
        kernel->nInputs = n_inputs;
        kernel->nOutputs = n_outputs;
        kernel->hash = hash;
        
        // Store sizes
        kernel->inputBytes = new size_t[n_inputs];
        kernel->outputBytes = new size_t[n_outputs];
        memcpy(kernel->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(kernel->outputBytes, output_sizes, n_outputs * sizeof(size_t));
        
        // Create IOSurfaces for inputs
        kernel->ioInputs = new IOSurfaceRef[n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            kernel->ioInputs[i] = ggml_ane_create_surface(input_sizes[i]);
            if (!kernel->ioInputs[i]) {
                GGML_ANE_LOG_ERROR("Failed to create input IOSurface %d", i);
                // TODO: cleanup
                return nullptr;
            }
        }
        
        // Create IOSurfaces for outputs
        kernel->ioOutputs = new IOSurfaceRef[n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            kernel->ioOutputs[i] = ggml_ane_create_surface(output_sizes[i]);
            if (!kernel->ioOutputs[i]) {
                GGML_ANE_LOG_ERROR("Failed to create output IOSurface %d", i);
                // TODO: cleanup
                return nullptr;
            }
        }
        
        // Build ANE request with IOSurface bindings
        NSMutableArray * inputObjects = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray * inputIndices = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            id ioObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->ioInputs[i]
            );
            [inputObjects addObject:ioObj];
            [inputIndices addObject:@(i)];
        }
        
        NSMutableArray * outputObjects = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray * outputIndices = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            id ioObj = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), kernel->ioOutputs[i]
            );
            [outputObjects addObject:ioObj];
            [outputIndices addObject:@(i)];
        }
        
        // Create the request
        kernel->request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            g_ANEReq, 
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inputObjects, inputIndices, 
            outputObjects, outputIndices, 
            nil, nil, @0
        );
        
        if (!kernel->request) {
            GGML_ANE_LOG_ERROR("Failed to create ANE request");
            // TODO: cleanup
            return nullptr;
        }
        
        // Cache if hash provided
        if (hash != 0) {
            std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
            g_kernel_cache[hash] = kernel;
            GGML_ANE_LOG_DEBUG("Kernel cached with hash 0x%016llx", hash);
        }
        
        GGML_ANE_LOG_INFO("ANE kernel created: %d inputs, %d outputs", n_inputs, n_outputs);
        return kernel;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Data Transfer
////////////////////////////////////////////////////////////////////////////////

void ggml_ane_write_input(ggml_ane_kernel_t kernel, int index, const void * data, size_t bytes) {
    if (!kernel || index < 0 || index >= kernel->nInputs) {
        GGML_ANE_LOG_ERROR("Invalid kernel or input index");
        return;
    }
    
    if (bytes > kernel->inputBytes[index]) {
        GGML_ANE_LOG_WARN("Input data size %zu > surface size %zu, truncating", 
                          bytes, kernel->inputBytes[index]);
        bytes = kernel->inputBytes[index];
    }
    
    IOSurfaceLock(kernel->ioInputs[index], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(kernel->ioInputs[index]), data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[index], 0, NULL);
}

void ggml_ane_read_output(ggml_ane_kernel_t kernel, int index, void * data, size_t bytes) {
    if (!kernel || index < 0 || index >= kernel->nOutputs) {
        GGML_ANE_LOG_ERROR("Invalid kernel or output index");
        return;
    }
    
    if (bytes > kernel->outputBytes[index]) {
        GGML_ANE_LOG_WARN("Output data size %zu > surface size %zu, truncating",
                          bytes, kernel->outputBytes[index]);
        bytes = kernel->outputBytes[index];
    }
    
    IOSurfaceLock(kernel->ioOutputs[index], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(kernel->ioOutputs[index]), bytes);
    IOSurfaceUnlock(kernel->ioOutputs[index], kIOSurfaceLockReadOnly, NULL);
}

////////////////////////////////////////////////////////////////////////////////
// Execution
////////////////////////////////////////////////////////////////////////////////

bool ggml_ane_eval_kernel(ggml_ane_kernel_t kernel) {
    if (!kernel || !kernel->model || !kernel->request) {
        GGML_ANE_LOG_ERROR("Invalid kernel state");
        return false;
    }
    
    fprintf(stderr, "[ANE] eval_kernel: nInputs=%d, nOutputs=%d\n", kernel->nInputs, kernel->nOutputs);
    
    @autoreleasepool {
        NSError * error = nil;
        
        CFAbsoluteTime evalStart = CFAbsoluteTimeGetCurrent();
        
        BOOL success = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            kernel->model, 
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, kernel->request, &error
        );
        
        double evalTime = (CFAbsoluteTimeGetCurrent() - evalStart) * 1000.0;
        
        if (!success) {
            fprintf(stderr, "[ANE ERROR] eval_kernel FAILED: %s\n", [[error description] UTF8String]);
            GGML_ANE_LOG_ERROR("ANE evaluation failed: %s", [[error description] UTF8String]);
            return false;
        }
        
        fprintf(stderr, "[ANE] eval_kernel SUCCESS in %.2f ms\n", evalTime);
        GGML_ANE_LOG_DEBUG("ANE kernel evaluated in %.2f ms", evalTime);
        return true;
    }
}

// Convenience function: write inputs, execute, read outputs
bool ggml_ane_execute(
    ggml_ane_kernel_t kernel,
    const void ** inputs,
    void ** outputs
) {
    if (!kernel) return false;
    
    // Write all inputs
    for (int i = 0; i < kernel->nInputs; i++) {
        if (inputs && inputs[i]) {
            ggml_ane_write_input(kernel, i, inputs[i], kernel->inputBytes[i]);
        }
    }
    
    // Execute
    if (!ggml_ane_eval_kernel(kernel)) {
        return false;
    }
    
    // Read all outputs
    for (int i = 0; i < kernel->nOutputs; i++) {
        if (outputs && outputs[i]) {
            ggml_ane_read_output(kernel, i, outputs[i], kernel->outputBytes[i]);
        }
    }
    
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Cleanup
////////////////////////////////////////////////////////////////////////////////

void ggml_ane_free_kernel(ggml_ane_kernel_t kernel) {
    if (!kernel) return;
    
    @autoreleasepool {
        NSError * error = nil;
        
        // Model will be released by ARC when kernel is deleted
        if (kernel->model) {
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                kernel->model, @selector(unloadWithQoS:error:), 21, &error
            );
        }
        
        // Request will be released by ARC
        
        // Release IOSurfaces
        if (kernel->ioInputs) {
            for (int i = 0; i < kernel->nInputs; i++) {
                if (kernel->ioInputs[i]) {
                    CFRelease(kernel->ioInputs[i]);
                }
            }
            delete[] kernel->ioInputs;
        }
        
        if (kernel->ioOutputs) {
            for (int i = 0; i < kernel->nOutputs; i++) {
                if (kernel->ioOutputs[i]) {
                    CFRelease(kernel->ioOutputs[i]);
                }
            }
            delete[] kernel->ioOutputs;
        }
        
        // Free size arrays
        delete[] kernel->inputBytes;
        delete[] kernel->outputBytes;
        
        // Remove from cache
        if (kernel->hash != 0) {
            std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
            g_kernel_cache.erase(kernel->hash);
        }
        
        // Cleanup temp directory
        if (kernel->tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:kernel->tmpDir error:nil];
        }
        
        delete kernel;
        GGML_ANE_LOG_DEBUG("ANE kernel freed");
    }
}

void ggml_ane_clear_cache(void) {
    std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
    
    for (auto & pair : g_kernel_cache) {
        ggml_ane_free_kernel(pair.second);
    }
    g_kernel_cache.clear();
    
    GGML_ANE_LOG_INFO("ANE kernel cache cleared");
}

////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////

size_t ggml_ane_get_kernel_input_size(ggml_ane_kernel_t kernel, int index) {
    if (!kernel || index < 0 || index >= kernel->nInputs) return 0;
    return kernel->inputBytes[index];
}

size_t ggml_ane_get_kernel_output_size(ggml_ane_kernel_t kernel, int index) {
    if (!kernel || index < 0 || index >= kernel->nOutputs) return 0;
    return kernel->outputBytes[index];
}

int ggml_ane_get_kernel_input_count(ggml_ane_kernel_t kernel) {
    return kernel ? kernel->nInputs : 0;
}

int ggml_ane_get_kernel_output_count(ggml_ane_kernel_t kernel) {
    return kernel ? kernel->nOutputs : 0;
}

#ifdef __cplusplus
}
#endif

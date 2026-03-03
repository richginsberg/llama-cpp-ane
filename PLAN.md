# ANE Backend for llama.cpp

**Goal:** Replace CoreML abstraction with direct `_ANEClient` API for 2-4× faster inference on Apple Silicon.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        llama.cpp                                 │
│                    (ggml graph execution)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ggml-backend-ane                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Buffer Type    │  │   Graph Plan    │  │    Device       │  │
│  │  (IOSurface)    │  │   (MIL Gen)     │  │   (ANE Dev)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AppleNeuralEngine.framework                     │
│              _ANEClient / _ANECompiler (Private)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANE Hardware (16 cores)                       │
│                    19 TFLOPS FP16 @ 2.8W                         │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation (20%)

**Goal:** Basic backend registration and buffer management

### Tasks:
- [ ] Create `ggml-ane/` directory structure
- [ ] Implement `ggml-ane.h` public header
- [ ] Implement `ggml-ane-device.h/mm` - Device enumeration and capabilities
- [ ] Implement `ggml-ane-buffer.h/mm` - IOSurface-backed buffer type
- [ ] Implement `ggml-ane.cpp` - Backend registration (`ggml_backend_reg_t`)
- [ ] Add CMakeLists.txt for ANE backend
- [ ] Unit test: Backend can be discovered and initialized

### Key Files to Create:
```
ggml/src/ggml-ane/
├── CMakeLists.txt
├── ggml-ane.h              # Public API
├── ggml-ane.cpp            # Backend registration
├── ggml-ane-device.h       # Device interface
├── ggml-ane-device.mm      # Device implementation (ObjC++)
├── ggml-ane-buffer.h       # Buffer interface
├── ggml-ane-buffer.mm      # IOSurface buffer implementation
└── ggml-ane-impl.h         # Internal structures
```

### Success Criteria:
- `ggml_backend_ane_reg()` returns valid registration
- Can allocate and free ANE buffers
- Buffers are IOSurface-backed for zero-copy

---

## Phase 2: ANE Runtime Wrapper (15%)

**Goal:** Wrap maderix's ANE runtime code for ggml integration

### Tasks:
- [ ] Port `ane_runtime.h` to `ggml-ane-runtime.h/mm`
- [ ] Add error handling and logging
- [ ] Implement kernel caching (avoid recompilation)
- [ ] Add thread safety (ANE is single-queue per process)
- [ ] Unit test: Can compile and execute simple MIL program

### Key Functions:
```cpp
// Initialize ANE framework
bool ggml_ane_init(void);

// Compile MIL + weights → kernel handle
ggml_ane_kernel_t ggml_ane_compile(
    const char *mil_text,
    const void *weights,
    size_t weights_size,
    int n_inputs, size_t *input_sizes,
    int n_outputs, size_t *output_sizes
);

// Execute kernel
bool ggml_ane_eval(ggml_ane_kernel_t kernel, void **inputs, void **outputs);

// Cleanup
void ggml_ane_kernel_free(ggml_ane_kernel_t kernel);
```

---

## Phase 3: MIL Compiler (25%)

**Goal:** Generate valid MIL from ggml operations

### Tasks:
- [ ] Port `ane_mil_gen.h` to `ggml-ane-mil.h/mm`
- [ ] Implement MIL generation for core ops:
  - [ ] `GGML_OP_MUL_MAT` → matmul or 1×1 conv
  - [ ] `GGML_OP_ADD` → elementwise add
  - [ ] `GGML_OP_MUL` → elementwise mul
  - [ ] `GGML_OP_RMS_NORM` → fused norm op
  - [ ] `GGML_OP_SOFT_MAX` → softmax
  - [ ] `GGML_OP_ROPE` → rope (or CPU fallback)
  - [ ] `GGML_OP_SILU` → elementwise silu
- [ ] Implement graph fusion for common patterns:
  - [ ] RMSNorm + Linear → fused conv
  - [ ] QKV projections → single 3-output kernel
  - [ ] FFN up (w1 + w3) → single 2-output kernel
- [ ] Implement weight blob builder
- [ ] Unit tests for each MIL generation function

### MIL Operation Mapping:

| GGML Op | MIL Op | Notes |
|---------|--------|-------|
| `MUL_MAT` | `matmul` or `conv` | Use 1×1 conv for 3× speedup |
| `ADD` | `add` | Elementwise |
| `MUL` | `mul` | Elementwise |
| `RMS_NORM` | Custom | reduce_sum + pow + mul |
| `SOFT_MAX` | `softmax` | Along axis |
| `ROPE` | CPU fallback | Complex, not in ANE native |
| `SILU` | `silu` | Elementwise |
| `SCALE` | `mul` | With scalar const |

### Weight Blob Format:
```
┌─────────────────────────────────┐
│ Global Header (64 bytes)        │
│ - magic: 0x01                   │
│ - version: 0x02                 │
├─────────────────────────────────┤
│ Chunk Header (64 bytes)         │
│ - magic: 0xDEADBEEF             │
│ - data_size                     │
│ - data_offset                   │
├─────────────────────────────────┤
│ FP16 Weights                    │
│ (row-major, [out_ch, in_ch])    │
└─────────────────────────────────┘
```

---

## Phase 4: Graph Execution (25%)

**Goal:** Execute ggml graphs on ANE

### Tasks:
- [ ] Implement `ggml_backend_ane_graph_plan_create()`
- [ ] Implement `ggml_backend_ane_graph_compute()`
- [ ] Implement operation support detection:
  - [ ] `ggml_backend_ane_supports_op()`
  - [ ] `ggml_backend_ane_offload_op()` - which ops to send to ANE
- [ ] Implement hybrid execution:
  - [ ] ANE for matrix ops (prefill)
  - [ ] CPU/SME for elementwise (decode, rope)
- [ ] Handle kernel cache and ~119 compile limit
- [ ] Implement graph optimization:
  - [ ] Op fusion
  - [ ] Memory reuse
- [ ] Integration tests with simple models

### Graph Plan Structure:
```cpp
struct ggml_backend_ane_graph_plan {
    // Compiled kernels for this graph
    ggml_ane_kernel_t *kernels;
    int n_kernels;
    
    // Execution order
    int *node_order;  // topological sort
    
    // Memory planning
    size_t total_io_memory;
    IOSurfaceRef *surfaces;
};
```

---

## Phase 5: Integration & Optimization (15%)

**Goal:** Production-ready integration with llama.cpp

### Tasks:
- [ ] Integrate with CMake build system
- [ ] Add `GGML_ANE` CMake option (default ON for Apple Silicon)
- [ ] Implement model weight conversion:
  - [ ] FP32 → FP16 for ANE
  - [ ] Weight blob packaging
- [ ] Add performance profiling:
  - [ ] Time per kernel
  - [ ] ANE utilization
  - [ ] Memory bandwidth
- [ ] Implement KV-cache management for ANE
- [ ] Benchmark against Metal backend:
  - [ ] Small models (1-3B)
  - [ ] Medium models (7-13B)
  - [ ] Large models (70B+ with offload)
- [ ] Documentation and examples

### Benchmark Targets:

| Model Size | Metal (t/s) | ANE Target (t/s) | Target Speedup |
|------------|-------------|------------------|----------------|
| 1B prefill | ~150 | ~300-450 | 2-3× |
| 7B prefill | ~40 | ~60-80 | 1.5-2× |
| 7B decode | ~25 | ~25-30 | 1-1.2× |
| Power (W) | ~15W | ~5W | 3× better |

---

## Testing Strategy

### Unit Tests (`tests/test-ggml-ane.cpp`):
1. Backend initialization
2. Buffer allocation/deallocation
3. MIL compilation
4. Single op execution (matmul, add, etc.)
5. Fused op execution
6. Graph plan creation
7. Graph execution

### Integration Tests:
1. Load small model (TinyLlama 1.1B)
2. Generate tokens
3. Compare output with CPU reference
4. Benchmark performance

### Test Commands:
```bash
# Build with ANE support
cmake -B build -DGGML_ANE=ON
cmake --build build

# Run unit tests
./build/bin/test-ggml-ane

# Run inference benchmark
./build/bin/llama-cli -m tinyllama.gguf -p "Hello" -n 100
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Private API changes | High | Pin to macOS 15.x, test extensively |
| ~119 compile limit | Medium | Implement exec() restart workaround |
| ANE doesn't support op | Medium | CPU fallback path |
| Memory management overhead | Medium | Prestacking, LRU warmup |
| IOSurface alignment | Low | Proper buffer padding |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Foundation | 3-4 days | None |
| Phase 2: Runtime Wrapper | 2-3 days | Phase 1 |
| Phase 3: MIL Compiler | 5-7 days | Phase 2 |
| Phase 4: Graph Execution | 5-7 days | Phase 3 |
| Phase 5: Integration | 3-4 days | Phase 4 |
| **Total** | **18-25 days** | |

---

## Progress Tracking

- **Started:** 2026-03-02
- **Current Phase:** Phase 1 (Foundation)
- **Overall Progress:** 0%

### Milestone Log:
- [ ] Phase 1 Complete - Backend registers successfully
- [ ] Phase 2 Complete - Can compile and run MIL
- [ ] Phase 3 Complete - All core ops have MIL generators
- [ ] Phase 4 Complete - Can execute simple graphs
- [ ] Phase 5 Complete - Benchmarking against Metal
- [ ] Production Ready - Merged and documented

---

## References

- [maderix/ANE](https://github.com/maderix/ANE) - Original ANE reverse engineering
- [Inside the M4 Apple Neural Engine](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) - Deep dive
- [ggml-metal](../ggml/src/ggml-metal/) - Reference backend implementation
- [Apple MLX](https://github.com/ml-explore/mlx) - Apple's ML framework

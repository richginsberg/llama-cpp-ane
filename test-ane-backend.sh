#!/bin/bash
# Test script to verify ANE backend is available

echo "=== Testing ANE Backend Registration ==="

# Test 1: Check if test binary exists
if [ -f "./build/bin/test-ggml-ane" ]; then
    echo "✓ test-ggml-ane binary found"
    ./build/bin/test-ggml-ane
else
    echo "✗ test-ggml-ane binary not found"
fi

echo ""
echo "=== Testing llama-cli with ANE ==="

# Test 2: List available backends
echo "Checking available backends..."
./build/bin/llama-cli --list-devices 2>&1 | head -20

echo ""
echo "=== Try running with ANE explicitly ==="
echo "Command: ./build/bin/llama-cli -m ~/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf -p 'Hello' -n 10 -ngl 0"
echo "(-ngl 0 disables GPU offload, should use CPU/ANE)"


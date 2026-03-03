# ggml-ane - Apple Neural Engine backend for ggml

cmake_minimum_required(VERSION 3.14)

option(GGML_ANE "Enable Apple Neural Engine backend" ON)

if (NOT GGML_ANE)
    message(STATUS "ANE backend disabled")
    return()
endif()

# Check for Apple Silicon
if (NOT APPLE)
    message(STATUS "ANE backend only available on Apple platforms")
    set(GGML_ANE OFF CACHE BOOL "Disable ANE on non-Apple" FORCE)
    return()
endif()

if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "ANE backend only available on Apple Silicon (arm64)")
    set(GGML_ANE OFF CACHE BOOL "Disable ANE on non-arm64" FORCE)
    return()
endif()

message(STATUS "ANE backend enabled")

# Find required frameworks
find_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)

# ANE backend sources
set(GGML_ANE_SOURCES
    ggml-ane.cpp
    ggml-ane-device.mm
    ggml-ane-buffer.mm
    ggml-ane-runtime.mm
    ggml-ane-mil.mm
    ggml-ane-ops.cpp
)

# Create static library
add_library(ggml-ane STATIC ${GGML_ANE_SOURCES})

target_include_directories(ggml-ane PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ggml-ane PRIVATE
    ${IOSURFACE_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREML_FRAMEWORK}
    ggml-base
)

target_compile_options(ggml-ane PRIVATE
    -Wall
    -Wextra
    -Werror=return-type
    -fobjc-arc
)

# Set C++ standard
set_target_properties(ggml-ane PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED ON
)

# Define build info
target_compile_definitions(ggml-ane PRIVATE
    GGML_ANE_BUILD
)

# Link with main ggml library
if (TARGET ggml)
    target_link_libraries(ggml PRIVATE ggml-ane)
    target_compile_definitions(ggml PRIVATE GGML_USE_ANE)
endif()

# Tests
if (GGML_BUILD_TESTS)
    add_executable(test-ggml-ane
        test-ggml-ane.cpp
    )
    target_link_libraries(test-ggml-ane PRIVATE ggml-ane)
    add_test(NAME test-ggml-ane COMMAND test-ggml-ane)
endif()

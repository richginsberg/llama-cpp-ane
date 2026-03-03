# ggml-ane - Apple Neural Engine backend for ggml

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)

message(STATUS "ANE frameworks found")

ggml_add_backend_library(ggml-ane
    ggml-ane.mm
    ggml-ane-device.mm
    ggml-ane-buffer.mm
    ggml-ane-runtime.mm
    ggml-ane-mil.mm
    ggml-ane-ops.cpp
)

target_link_libraries(ggml-ane PRIVATE
    ${FOUNDATION_FRAMEWORK}
    ${IOSURFACE_FRAMEWORK}
    ${COREML_FRAMEWORK}
)

target_compile_options(ggml-ane PRIVATE
    -Wall
    -Wextra
    -Werror=return-type
    -fobjc-arc
)

# Enable debug logging if requested
if (GGML_ANE_DEBUG)
    target_compile_definitions(ggml-ane PRIVATE GGML_ANE_DEBUG)
endif()

# Simple matmul test (build manually, not through GGML_BUILD_TESTS)
if (BUILD_SHARED_LIBS)
    add_executable(test-ane-matmul test-ane-matmul.cpp)
    target_link_libraries(test-ane-matmul PRIVATE ggml-ane)
    target_include_directories(test-ane-matmul PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(test-ane-matmul PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    )
endif()

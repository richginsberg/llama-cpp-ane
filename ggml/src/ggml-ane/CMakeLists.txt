# ggml-ane - Apple Neural Engine backend for ggml

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)

message(STATUS "ANE frameworks found")

ggml_add_backend_library(ggml-ane
    ggml-ane.cpp
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
    add_compile_definitions(GGML_ANE_DEBUG)
endif()

# Link with main ggml library
target_link_libraries(ggml PRIVATE ggml-ane)
target_compile_definitions(ggml PRIVATE GGML_USE_ANE)

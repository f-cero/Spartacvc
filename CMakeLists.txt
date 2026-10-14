# Spartacvς — CMake build

cmake_minimum_required(VERSION 3.16)
project(spartacvc C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# ONNX Runtime
# ------------------------------------------------------------------
# Provide ORT_ROOT as a cache variable pointing to the extracted
# onnxruntime release (e.g. onnxruntime-win-x64-1.20.1).
#
#   cmake -DORT_ROOT=third_party/onnxruntime/onnxruntime-win-x64-1.20.1 ..
#
# If not set, try FetchContent (downloads from GitHub releases).
if(NOT DEFINED ORT_ROOT)
    include(FetchContent)
    set(ORT_VERSION "1.20.1")
    if(WIN32)
        set(ORT_PLATFORM "win-x64")
    else()
        set(ORT_PLATFORM "linux-x64")
    endif()
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tar.gz")
    FetchContent_Declare(onnxruntime
        URL ${ORT_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")
endif()

message(STATUS "ONNX Runtime: ${ORT_ROOT}")

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(SPCV_DEBUG "Enable debug tracing" OFF)

# ------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------
set(SRC
    src/onnx.c
    src/debug.c
    src/mel.c
    src/f0.c
    src/model.c
    src/onnx_init.c
    src/main.c
)

add_executable(spartacvc ${SRC})

target_include_directories(spartacvc PRIVATE
    src
    ${ORT_ROOT}/include
)

if(SPCV_DEBUG)
    target_compile_definitions(spartacvc PRIVATE SPCV_DEBUG)
endif()

# ------------------------------------------------------------------
# Link ONNX Runtime
# ------------------------------------------------------------------
if(WIN32)
    target_link_libraries(spartacvc PRIVATE
        ${ORT_ROOT}/lib/onnxruntime.lib
    )
    # Copy DLLs next to the exe
    add_custom_command(TARGET spartacvc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_ROOT}/lib/onnxruntime.dll
            ${CMAKE_BINARY_DIR}/
    )
    add_custom_command(TARGET spartacvc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_ROOT}/lib/onnxruntime_providers_shared.dll
            ${CMAKE_BINARY_DIR}/
    )
else()
    target_link_libraries(spartacvc PRIVATE
        ${ORT_ROOT}/lib/libonnxruntime.so
    )
    # rpath so the .so is found at runtime
    set_target_properties(spartacvc PROPERTIES
        BUILD_RPATH "${ORT_ROOT}/lib"
        INSTALL_RPATH "${ORT_ROOT}/lib"
    )
endif()

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install(TARGETS spartacvc RUNTIME DESTINATION bin)

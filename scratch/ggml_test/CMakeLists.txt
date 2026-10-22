cmake_minimum_required(VERSION 3.18)
project(ggml_test C CXX)

# Reference kokoro's ggml source checkout (no copy).
set(GGML_SRC "X:/pi-code/tts/kokoro.cpp/ggml")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)

# Target the PG500 (sm_70) + a couple common ones.
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "" FORCE)

set(GGML_BUILD_TESTS OFF)
set(GGML_BUILD_EXAMPLES OFF)
set(GGML_CPU ON)
set(GGML_CUDA ON)
set(GGML_VULKAN OFF)
set(GGML_METAL OFF)
set(GGML_SYCL OFF)
set(GGML_RPC OFF)
set(GGML_OPENCL OFF)

add_subdirectory("${GGML_SRC}" "${CMAKE_CURRENT_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

add_executable(ggml_test test.c)
target_link_libraries(ggml_test PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(ggml_test PRIVATE
    "${GGML_SRC}/include"
    "${GGML_SRC}/src")

# --- in-app .pth -> GGUF conversion test ---
set(SPCV "X:/pi-code/spartacvc/src")
add_executable(test_pth2gguf
    test_pth2gguf.c
    "${SPCV}/pth2gguf.c"
    "${SPCV}/pth_loader.c"
    "${SPCV}/zip_reader.c"
    "${SPCV}/pickle.c")
target_link_libraries(test_pth2gguf PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_pth2gguf PRIVATE
    "${GGML_SRC}/include"
    "${SPCV}")

# --- gguf load + alloc minimal ---
add_executable(test_gguf_load_min test_gguf_load_min.c)
target_link_libraries(test_gguf_load_min PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_gguf_load_min PRIVATE "${GGML_SRC}/include")

# --- minimal CPU backend probe ---
add_executable(test_cpu_min test_cpu_min.c)
target_link_libraries(test_cpu_min PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_cpu_min PRIVATE "${GGML_SRC}/include")

# --- ggml RVC conv1d test (GGUF -> CUDA -> conv_1d vs pure C) ---
add_executable(test_rvc_conv
    test_rvc_conv.c
    "${SPCV}/gguf_load.c")
target_link_libraries(test_rvc_conv PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_rvc_conv PRIVATE
    "${GGML_SRC}/include"
    "${SPCV}")

add_executable(test_layout test_layout.c)
target_link_libraries(test_layout PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_layout PRIVATE "${GGML_SRC}/include")

add_executable(dump_gguf dump_gguf.c)
target_link_libraries(dump_gguf PRIVATE ggml)
target_include_directories(dump_gguf PRIVATE "${GGML_SRC}/include")

add_executable(test_rvc_ops
    test_rvc_ops.c
    "${SPCV}/gguf_load.c")
target_link_libraries(test_rvc_ops PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_rvc_ops PRIVATE
    "${GGML_SRC}/include"
    "${SPCV}")

add_executable(test_rvc_enc
    test_rvc_enc.c
    "${SPCV}/gguf_load.c"
    "${SPCV}/ggml_rvc_enc.c"
    "${SPCV}/rvc_infer.c"
    "${SPCV}/rvc.c"
    "${SPCV}/pth_loader.c"
    "${SPCV}/zip_reader.c"
    "${SPCV}/pickle.c"
    "${SPCV}/threads.c")
target_link_libraries(test_rvc_enc PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_rvc_enc PRIVATE
    "${GGML_SRC}/include"
    "${SPCV}")

add_executable(test_rvc_enc_gpu
    test_rvc_enc_gpu.c
    "${SPCV}/gguf_load.c"
    "${SPCV}/rvc_infer.c"
    "${SPCV}/rvc.c"
    "${SPCV}/pth_loader.c"
    "${SPCV}/zip_reader.c"
    "${SPCV}/pickle.c"
    "${SPCV}/threads.c")
target_link_libraries(test_rvc_enc_gpu PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_rvc_enc_gpu PRIVATE
    "${GGML_SRC}/include"
    "${SPCV}")

add_executable(debug_mulmat debug_mulmat.c "${SPCV}/gguf_load.c")
target_link_libraries(debug_mulmat PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(debug_mulmat PRIVATE "${GGML_SRC}/include" "${SPCV}")

add_executable(test_enc_min test_enc_min.c "${SPCV}/gguf_load.c")
target_link_libraries(test_enc_min PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_enc_min PRIVATE "${GGML_SRC}/include" "${SPCV}")

add_executable(test_enc_min2 test_enc_min2.c "${SPCV}/gguf_load.c")
target_link_libraries(test_enc_min2 PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_enc_min2 PRIVATE "${GGML_SRC}/include" "${SPCV}")

add_executable(gen_ref gen_ref.c "${SPCV}/rvc_infer.c" "${SPCV}/rvc.c" "${SPCV}/pth_loader.c" "${SPCV}/zip_reader.c" "${SPCV}/pickle.c" "${SPCV}/threads.c")
target_link_libraries(gen_ref PRIVATE ggml ggml-cpu)
target_include_directories(gen_ref PRIVATE "${GGML_SRC}/include" "${SPCV}")

add_executable(test_rvc_flow_gpu test_rvc_flow_gpu.c "${SPCV}/gguf_load.c")
target_link_libraries(test_rvc_flow_gpu PRIVATE ggml ggml-cpu ggml-cuda)
target_include_directories(test_rvc_flow_gpu PRIVATE "${GGML_SRC}/include" "${SPCV}")

add_executable(gen_flow_ref gen_flow_ref.c "${SPCV}/rvc_infer.c" "${SPCV}/rvc.c" "${SPCV}/pth_loader.c" "${SPCV}/zip_reader.c" "${SPCV}/pickle.c" "${SPCV}/threads.c")
target_link_libraries(gen_flow_ref PRIVATE ggml ggml-cpu)
target_include_directories(gen_flow_ref PRIVATE "${GGML_SRC}/include" "${SPCV}")

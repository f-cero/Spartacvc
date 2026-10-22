@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE="C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
%CMAKE% -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_CPU=ON -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_RPC=OFF -DGGML_OPENCL=OFF >nul 2>&1
%CMAKE% --build build --config Release --target test_e2e_gpu --parallel
echo BUILD=%ERRORLEVEL%

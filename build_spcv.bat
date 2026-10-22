@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d X:\pi-code\spartacvc
set ORT=third_party\onnxruntime
set GGML_INC=X:\pi-code\tts\kokoro.cpp\ggml\include
set GGML_LIB=X:\pi-code\spartacvc\scratch\ggml_test\build\ggml\src\Release
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64
cl /O2 /MD /W3 /D_CRT_SECURE_NO_WARNINGS /DSPCV_DEBUG /I src /I "%ORT%\include" /I "%GGML_INC%" /Fe:spartacvc.exe src/onnx.c src/debug.c src/threads.c src/mel.c src/f0.c src/model.c src/onnx_init.c src/zip_reader.c src/pickle.c src/pth_loader.c src/pth2gguf.c src/gguf_check.c src/gguf_load.c src/ggml_pipeline.c src/rvc.c src/rvc_infer.c src/faiss_index.c src/gpu_list.c src/backends.c src/main.c /link /SUBSYSTEM:CONSOLE /STACK:4194304 dxgi.lib d3d12.lib ole32.lib advapi32.lib "%GGML_LIB%\ggml-base.lib" "%GGML_LIB%\ggml-cpu.lib" "X:\pi-code\spartacvc\scratch\ggml_test\build\ggml\src\ggml-cuda\Release\ggml-cuda.lib" "%CUDA_LIB%\cublas.lib" "%CUDA_LIB%\cublasLt.lib" "%CUDA_LIB%\cudart.lib" "%CUDA_LIB%\cufft.lib" "%CUDA_LIB%\curand.lib" "%CUDA_LIB%\cuda.lib" 2>&1
echo EXIT=%ERRORLEVEL%

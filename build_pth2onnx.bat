@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

echo Building pth2onnx...
cl /O2 /W3 /nologo /I src ^
   src\onnx_emit.c ^
   src\rvc_graph.c ^
   src\rvc_graph_bin.c ^
   src\pth2onnx.c ^
   src\pth_loader.c ^
   src\pickle.c ^
   src\zip_reader.c ^
   /Fe:pth2onnx.exe

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD OK
endlocal

@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

echo Building pth2onnx (DEBUG)...
cl /O2 /W3 /nologo /DSPCV_DEBUG /I src ^
   src\onnx_emit.c ^
   src\rvc_graph.c ^
   src\rvc_graph_bin.c ^
   src\pth2onnx.c ^
   src\pth_loader.c ^
   src\pickle.c ^
   src\zip_reader.c ^
   src\debug.c ^
   /Fe:pth2onnx.exe

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)
echo BUILD OK

echo === Running pth2onnx ===
set SPCV_DEBUG=1
pth2onnx.exe C:\temp\nahida_extracted\nahida-jp.pth C:\temp\nahida_generated.onnx
echo RC=%ERRORLEVEL%
endlocal

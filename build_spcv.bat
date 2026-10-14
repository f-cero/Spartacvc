@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d X:\pi-code\spartacvc
set ORT=third_party\onnxruntime\onnxruntime-win-x64-1.20.1
cl /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /DSPCV_DEBUG /I src /I "%ORT%\include" /Fe:spartacvc.exe src/onnx.c src/debug.c src/mel.c src/f0.c src/model.c src/onnx_init.c src/main.c /link "%ORT%\lib\onnxruntime.lib" /SUBSYSTEM:CONSOLE 2>&1
echo EXIT=%ERRORLEVEL%

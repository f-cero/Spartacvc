@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE="C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
%CMAKE% --build build --config Release --target test_rvc_enc_gpu --parallel
echo BUILD=%ERRORLEVEL%

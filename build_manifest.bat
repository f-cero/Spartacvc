@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d X:\pi-code\spartacvc
cl /O2 /W3 /Fe:test_manifest.exe test_manifest.c src\manifest.c /link /SUBSYSTEM:CONSOLE 2>&1
echo EXIT=%ERRORLEVEL%

@echo off
setlocal
echo ============================================================
echo   Spartacvs - MSVC Spectre-mitigated libs (elevated)
echo   (required to build the KMDF audio driver)
echo ============================================================
echo.
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Run this as ADMINISTRATOR.
  pause & exit /b 1
)
echo [OK] Running elevated.
echo.
rem VS processes block the installer (pre-check VSProcessesRunning, exit 8006)
tasklist /FI "IMAGENAME eq devenv.exe" 2>nul | findstr /I "devenv" >nul
if %errorlevel% equ 0 (
  echo [WARN] Visual Studio is running - closing it
  taskkill /IM devenv.exe /T /F >nul 2>&1
)
taskkill /IM VBCSCompiler.exe /F >nul 2>&1
taskkill /IM MSMDLauncher.exe /F >nul 2>&1
ping -n 3 127.0.0.1 >nul
set "VSI=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
echo Adding Spectre-mitigated CRT libs (x86/x64) for MSVC 14.44 ...
"%VSI%" modify --installPath "C:\Program Files\Microsoft Visual Studio\2022\Professional" --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64.Spectre --passive --norestart
set E=%errorlevel%
echo.
echo vs_installer exit code: %E%
echo.
echo === verify Spectre libs landed ===
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\lib\Spectre\onecore\x64" (
  echo   [OK] MSVC 14.44.35207 : Spectre\onecore\x64 present
) else (
  echo   [--] MSVC 14.44.35207 : Spectre\onecore\x64 NOT present
)
echo.
echo ============================================================
echo   Done. Tell pi to continue when it finishes.
echo ============================================================
pause
endlocal

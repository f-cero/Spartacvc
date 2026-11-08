@echo off
setlocal
echo ============================================================
echo   Spartacv\u03C2 - Windows Driver Kit 11 installer (elevated)
echo ============================================================
echo.
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] This script must be run as ADMINISTRATOR.
  echo         Right-click -^> "Run as administrator".
  pause & exit /b 1
)
echo [OK] Running elevated.
echo.

echo [1/3] Installing WDK 11 (10.0.22000) - this downloads ~1.5 GB...
cd /d C:\temp
if not exist wdksetup.exe (
  echo       downloading wdksetup.exe ...
  curl -sL -o wdksetup.exe "https://download.microsoft.com/download/7/d/6/7d602355-8ae9-414c-ae36-109ece2aade6/wdk/wdksetup.exe"
)
wdksetup.exe /features + /quiet /norestart /log "C:\temp\wdk_elev.log"
set WEXIT=%errorlevel%
echo       wdksetup exit code: %WEXIT%
echo.

echo [2/3] Verifying WDK tools...
set FOUND=
for %%V in (10.0.22000.0 10.0.26100.0) do (
  if exist "C:\Program Files (x86)\Windows Kits\10\bin\%%V\x86\inf2cat.exe" set FOUND=%%V
)
if defined FOUND (
  echo       [OK] WDK found at version %FOUND%
) else (
  echo       [WARN] inf2cat not found in expected dirs. Searching...
  where /r "C:\Program Files (x86)\Windows Kits" inf2cat.exe
)
echo.

echo [3/3] Checking signtool...
where /r "C:\Program Files (x86)\Windows Kits" signtool.exe
echo.
echo ============================================================
echo   Done. WDK install exit code was: %WEXIT%
echo   Re-run this if it failed. Then tell pi to continue.
echo ============================================================
pause
endlocal

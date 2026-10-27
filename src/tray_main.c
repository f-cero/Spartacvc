// tray_main.c — Win32 GUI entry point for spartacvc tray mode.
// Build with /SUBSYSTEM:WINDOWS (no console).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "tray.h"

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpCmdLine, int nShow) {
    (void)hInst; (void)hPrev; (void)lpCmdLine; (void)nShow;
    return spcv_tray_run();
}

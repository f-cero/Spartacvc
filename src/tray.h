// tray.h — Windows system tray entry point for spartacvc.
//
// Builds a Win32 GUI app (no console) that:
//   - Shows a tray icon
//   - Enumerates available backends (CPU, CUDA, Vulkan) on load
//   - Right-click menu: select backend, start/stop, quit
//   - Runs the RVC pipeline in a worker thread when started
//
// Link with /SUBSYSTEM:WINDOWS and entry point WinMain.
#ifndef SPCV_TRAY_H
#define SPCV_TRAY_H

#ifdef __cplusplus
extern "C" {
#endif

/** Win32 entry point. Call from WinMain. */
int spcv_tray_run(void);

#ifdef __cplusplus
}
#endif

#endif // SPCV_TRAY_H

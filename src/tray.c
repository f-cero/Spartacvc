// tray.c — Windows system tray for spartacvc.
//
// Enumerates available compute backends on load, shows a tray icon,
// and provides a right-click menu to select the active pipeline.
//
// Backends are probed at startup:
//   - CPU (always available)
//   - CUDA (if NVIDIA device present)
//   - Vulkan (if Vulkan device present)
//
// The selected backend is stored and can be queried by the pipeline code.
//
// Win32 APIs used: Shell_NotifyIcon, TrackPopupMenu, RegisterClass.
// No external dependencies beyond Win32 + ggml backend headers.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <string.h>
#include "tray.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-vulkan.h"

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "user32.lib")

// --- Menu item IDs ---
#define ID_MNU_BASE      100
#define ID_MNU_SEP1      200
#define ID_MNU_START     201
#define ID_MNU_STOP      202
#define ID_MNU_SEP2      203
#define ID_MNU_QUIT      204

// --- Backend info ---
typedef struct {
    const wchar_t *name;    // display name (wide/UTF-16)
    int         be_id;      // SPCV_BE_* enum value
    int         available;  // 1 if device found
    int         n_devices;  // number of devices
    wchar_t     desc[128];  // device description
} backend_info_t;

#define MAX_BACKENDS 8

static backend_info_t g_backends[MAX_BACKENDS];
static int            g_n_backends = 0;
static int            g_selected   = 0;   // index into g_backends[]
static int            g_running    = 0;   // pipeline running?

static NOTIFYICONDATAW g_nid = {0};
static HWND            g_hwnd = NULL;
static HICON           g_hicon = NULL;

// --- Backend enumeration ---

static void probe_backends(void) {
    g_n_backends = 0;

    // CPU (always)
    g_backends[g_n_backends].name = L"CPU";
    g_backends[g_n_backends].be_id = 0;
    g_backends[g_n_backends].available = 1;
    g_backends[g_n_backends].n_devices = 1;
    wcscpy_s(g_backends[g_n_backends].desc, 128, L"General Purpose");
    g_n_backends++;

    // CUDA
    int n_cuda = ggml_backend_cuda_get_device_count();
    if (n_cuda > 0) {
        g_backends[g_n_backends].name = L"CUDA";
        g_backends[g_n_backends].be_id = 1;
        g_backends[g_n_backends].available = 1;
        g_backends[g_n_backends].n_devices = n_cuda;
        wcscpy_s(g_backends[g_n_backends].desc, 128, L"NVIDIA GPU");
        g_n_backends++;
    }

    // Vulkan
    int n_vk = ggml_backend_vk_get_device_count();
    if (n_vk > 0) {
        g_backends[g_n_backends].name = L"Vulkan";
        g_backends[g_n_backends].be_id = 3;
        g_backends[g_n_backends].available = 1;
        g_backends[g_n_backends].n_devices = n_vk;
        char desc_n[128] = {0};
        ggml_backend_vk_get_device_description(0, desc_n, sizeof(desc_n));
        MultiByteToWideChar(CP_UTF8, 0, desc_n, -1, g_backends[g_n_backends].desc, 128);
        g_n_backends++;
    }

    // Default selection: first GPU if available, else CPU
    if (g_n_backends > 1) g_selected = g_n_backends - 1; // prefer last (Vulkan)
    else if (g_n_backends > 0) g_selected = 0;
}

// --- Tray icon ---

static int add_tray_icon(void) {
    ZeroMemory(&g_nid, sizeof(g_nid));
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_APP + 1;
    g_nid.hIcon = g_hicon;
    wcscpy_s(g_nid.szTip, 128, L"Spartacv\u03C2");
    return Shell_NotifyIconW(NIM_ADD, &g_nid);
}

static void remove_tray_icon(void) {
    Shell_NotifyIconW(NIM_DELETE, &g_nid);
}

// --- Context menu ---

static void show_menu(void) {
    HMENU hmenu = CreatePopupMenu();
    if (!hmenu) return;

    POINT pt;
    GetCursorPos(&pt);

    // Backend selection
    for (int i = 0; i < g_n_backends; i++) {
        if (!g_backends[i].available) continue;
        UINT id = ID_MNU_BASE + i;
        UINT flags = MF_STRING | (i == g_selected ? MF_CHECKED : MF_UNCHECKED);
        // Build label: "CUDA — NVIDIA GPU (1)"
        wchar_t label[256];
        swprintf(label, 256, L"%s — %s (%d)",
                 g_backends[i].name, g_backends[i].desc, g_backends[i].n_devices);
        AppendMenuW(hmenu, flags, (UINT_PTR)id, label);
    }

    // Separator
    AppendMenuW(hmenu, MF_SEPARATOR, 0, NULL);

    // Start / Stop
    if (g_running) {
        AppendMenuW(hmenu, MF_STRING, ID_MNU_STOP, L"Stop");
    } else {
        AppendMenuW(hmenu, MF_STRING, ID_MNU_START, L"Start");
    }

    // Separator + Quit
    AppendMenuW(hmenu, MF_SEPARATOR, 0, NULL);
    AppendMenuW(hmenu, MF_STRING, ID_MNU_QUIT, L"Quit");

    SetForegroundWindow(g_hwnd);
    int cmd = TrackPopupMenu(hmenu, TPM_RIGHTBUTTON | TPM_RETURNCMD,
                             pt.x, pt.y, 0, g_hwnd, NULL);
    DestroyMenu(hmenu);

    if (cmd >= ID_MNU_BASE && cmd < ID_MNU_BASE + MAX_BACKENDS) {
        g_selected = cmd - ID_MNU_BASE;
        // Update tooltip to show selected backend
        wchar_t tip[128];
        swprintf(tip, 128, L"Spartacv\u03C2 [%s]", g_backends[g_selected].name);
        wcscpy_s(g_nid.szTip, 128, tip);
        Shell_NotifyIconW(NIM_MODIFY, &g_nid);
    } else if (cmd == ID_MNU_QUIT) {
        PostMessageW(g_hwnd, WM_CLOSE, 0, 0);
    }
    // ID_MNU_START / ID_MNU_STOP: TODO — launch pipeline thread
}

// --- Window proc ---

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_APP + 1:
        if (lp == WM_RBUTTONUP || lp == WM_LBUTTONUP) {
            show_menu();
        }
        return 0;
    case WM_CLOSE:
        remove_tray_icon();
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// --- Main entry ---

int spcv_tray_run(void) {
    // Probe backends
    probe_backends();

    // Create hidden window
    WNDCLASSEXW wc = {0};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"SpartacvcTray";
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(0, L"SpartacvcTray", L"Spartacv\u03C2",
                             0, 0, 0, 0, 0,
                             HWND_MESSAGE, NULL, NULL, NULL);
    if (!g_hwnd) return 1;

    // Load default icon (app icon or system hand)
    g_hicon = LoadIconW(NULL, IDI_APPLICATION);

    // Add tray icon
    add_tray_icon();

    // Message loop
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Cleanup
    if (g_hicon) DestroyIcon(g_hicon);
    return 0;
}

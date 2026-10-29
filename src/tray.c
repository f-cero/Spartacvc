// tray.c — Windows system tray for spartacvc.
//
// Menu:
//   Pipeline:  ✓ Vulkan / CUDA / CPU
//   Model:     current model + directory listing + Browse
//   Audio:     In/Out devices + Change
//   Start/Stop
//   Convert file...
//   Quit
//
// All strings wide/UTF-16. No console window.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <commdlg.h>
#include <shlobj.h>
#include <devpropdef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tray.h"
#include "wasapi.h"
#include "pipeline_rt.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-vulkan.h"

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "comctl32.lib")

// --- Menu item IDs ---
#define ID_BASE          100   // backend selection: ID_BASE + i
#define ID_MODEL_BASE    300   // model selection: ID_MODEL_BASE + i
#define ID_MODEL_BROWSE  390
#define ID_DEV_IN_BASE   400   // input device: ID_DEV_IN_BASE + i
#define ID_DEV_OUT_BASE  500   // output device: ID_DEV_OUT_BASE + i
#define ID_START         600
#define ID_STOP          601
#define ID_CONVERT       602
#define ID_QUIT          603

// --- State ---
typedef struct {
    const wchar_t *name;
    int  be_id;
    int  available;
    int  n_devices;
    wchar_t desc[128];
} backend_info_t;

#define MAX_BACKENDS  8
#define MAX_MODELS    32
#define MAX_DEVS      16

static backend_info_t g_backends[MAX_BACKENDS];
static int  g_n_backends = 0;
static int  g_sel_be     = 0;

static wchar_t g_models[MAX_MODELS][260];
static int     g_n_models = 0;
static int     g_sel_model = 0;
static wchar_t g_model_dir[260] = L"models\\voices";

static wasapi_dev_t g_dev_in[MAX_DEVS];
static int  g_n_dev_in = 0;
static int  g_sel_in = 0;

static wasapi_dev_t g_dev_out[MAX_DEVS];
static int  g_n_dev_out = 0;
static int  g_sel_out = 0;

static int  g_running = 0;

static void save_config(void);

static NOTIFYICONDATAW g_nid = {0};
static HWND   g_hwnd  = NULL;
static HICON  g_hicon = NULL;

// --- Backend probe ---
static void probe_backends(void) {
    g_n_backends = 0;
    g_backends[g_n_backends].name = L"CPU";
    g_backends[g_n_backends].be_id = 0;
    g_backends[g_n_backends].available = 1;
    g_backends[g_n_backends].n_devices = 1;
    wcscpy_s(g_backends[g_n_backends].desc, 128, L"General Purpose");
    g_n_backends++;

    int n = ggml_backend_cuda_get_device_count();
    if (n > 0) {
        g_backends[g_n_backends].name = L"CUDA";
        g_backends[g_n_backends].be_id = 1;
        g_backends[g_n_backends].available = 1;
        g_backends[g_n_backends].n_devices = n;
        wcscpy_s(g_backends[g_n_backends].desc, 128, L"NVIDIA GPU");
        g_n_backends++;
    }
    n = ggml_backend_vk_get_device_count();
    if (n > 0) {
        g_backends[g_n_backends].name = L"Vulkan";
        g_backends[g_n_backends].be_id = 3;
        g_backends[g_n_backends].available = 1;
        g_backends[g_n_backends].n_devices = n;
        char dn[128] = {0};
        ggml_backend_vk_get_device_description(0, dn, sizeof(dn));
        MultiByteToWideChar(CP_UTF8, 0, dn, -1, g_backends[g_n_backends].desc, 128);
        g_n_backends++;
    }
    g_sel_be = (g_n_backends > 1) ? g_n_backends - 1 : 0;
}

// --- Model scan ---
static void scan_models(void) {
    g_n_models = 0;
    wchar_t pattern[300];
    swprintf(pattern, 300, L"%s\\*.gguf", g_model_dir);
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pattern, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            wcscpy_s(g_models[g_n_models], 260, fd.cFileName);
            g_n_models++;
        } while (FindNextFileW(h, &fd) && g_n_models < MAX_MODELS);
        FindClose(h);
    }
    // Also scan for .pth
    swprintf(pattern, 300, L"%s\\*.pth", g_model_dir);
    h = FindFirstFileW(pattern, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
            wcscpy_s(g_models[g_n_models], 260, fd.cFileName);
            g_n_models++;
        } while (FindNextFileW(h, &fd) && g_n_models < MAX_MODELS);
        FindClose(h);
    }
    g_sel_model = 0;
}

// --- Audio device scan ---
static void scan_devices(void) {
    wasapi_co_init();
    g_n_dev_in = wasapi_enum_inputs(g_dev_in, MAX_DEVS);
    g_n_dev_out = wasapi_enum_outputs(g_dev_out, MAX_DEVS);
    g_sel_in = 0;
    g_sel_out = 0;
}

// --- Convert file (call spartacvc.exe CLI) ---
static void do_convert_file(void) {
    if (g_n_models == 0) {
        MessageBoxW(g_hwnd, L"No models found.\nPlace .gguf or .pth files in models\\voices\\",
                    L"Spartacv\u03C2", MB_ICONWARNING);
        return;
    }

    wchar_t in_file[260] = L"";
    wchar_t out_file[260] = L"";

    OPENFILENAMEW ofn = { sizeof(ofn) };
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFilter = L"WAV files\0*.wav\0All files\0*.*\0";
    ofn.lpstrFile = in_file;
    ofn.nMaxFile = 260;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!GetOpenFileNameW(&ofn)) return;

    ofn.lpstrFile = out_file;
    ofn.lpstrDefExt = L"wav";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
    if (!GetSaveFileNameW(&ofn)) return;

    // Full model path
    wchar_t model_path[300];
    swprintf(model_path, 300, L"%s\\%s", g_model_dir, g_models[g_sel_model]);

    const wchar_t *be = (g_backends[g_sel_be].be_id == 1) ? L"cuda"
                       : (g_backends[g_sel_be].be_id == 3) ? L"vulkan" : L"cpu";

    // Find spartacvc.exe next to this exe
    wchar_t exe_dir[260];
    GetModuleFileNameW(NULL, exe_dir, 260);
    wchar_t *slash = wcsrchr(exe_dir, L'\\');
    if (slash) *(slash + 1) = 0;
    wchar_t cli_path[300];
    swprintf(cli_path, 300, L"%sspartacvc.exe", exe_dir);

    // Hubert + RMVPE paths (relative to exe dir)
    wchar_t hubert[300], rmvpe[300];
    swprintf(hubert, 300, L"%smodels\base\vec-768-layer-12.onnx", exe_dir);
    swprintf(rmvpe, 300, L"%smodels\base\rmvpe.onnx", exe_dir);

    wchar_t cmd[2048];
    swprintf(cmd, 2048,
        L"\"%s\" \"%s\" --voice \"%s\" --hubert \"%s\" --rmvpe \"%s\" --backend %s --out \"%s\"",
        cli_path, in_file, model_path, hubert, rmvpe, be, out_file);

    // Show progress
    MessageBoxW(g_hwnd, L"Converting...\nThis may take a minute.", L"Spartacv\u03C2", MB_OK);

    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = {0};
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                        exe_dir, NULL, &si, &pi)) {
        MessageBoxW(g_hwnd, L"Failed to launch spartacvc.exe",
                    L"Spartacv\u03C2", MB_ICONERROR);
        return;
    }
    WaitForSingleObject(pi.hProcess, 120000);
    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    if (exit_code == 0)
        MessageBoxW(g_hwnd, L"Conversion complete.", L"Spartacv\u03C2", MB_OK | MB_ICONINFORMATION);
    else
        swprintf(cmd, 2048, L"Conversion failed (exit code %lu).", exit_code),
        MessageBoxW(g_hwnd, cmd, L"Spartacv\u03C2", MB_ICONERROR);
}

// --- Tray icon ---
static void add_tray_icon(void) {
    ZeroMemory(&g_nid, sizeof(g_nid));
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_APP + 1;
    g_nid.hIcon = g_hicon;
    wcscpy_s(g_nid.szTip, 128, L"Spartacv\u03C2");
    Shell_NotifyIconW(NIM_ADD, &g_nid);
}

static void update_tip(void) {
    wchar_t tip[128];
    swprintf(tip, 128, L"Spartacv\u03C2 [%s]%s",
             g_backends[g_sel_be].name, g_running ? L" \u25CF" : L"");
    wcscpy_s(g_nid.szTip, 128, tip);
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

// --- Context menu ---
static void show_menu(void) {
    HMENU m = CreatePopupMenu();
    POINT pt; GetCursorPos(&pt);

    // Pipeline
    for (int i = 0; i < g_n_backends; i++) {
        if (!g_backends[i].available) continue;
        wchar_t lbl[256];
        swprintf(lbl, 256, L"%s \u2014 %s", g_backends[i].name, g_backends[i].desc);
        AppendMenuW(m, MF_STRING | (i == g_sel_be ? MF_CHECKED : 0),
                    ID_BASE + i, lbl);
    }
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);

    // Model
    if (g_n_models > 0) {
        for (int i = 0; i < g_n_models; i++) {
            AppendMenuW(m, MF_STRING | (i == g_sel_model ? MF_CHECKED : 0),
                        ID_MODEL_BASE + i, g_models[i]);
        }
        AppendMenuW(m, MF_STRING, ID_MODEL_BROWSE, L"Browse...");
    } else {
        AppendMenuW(m, MF_GRAYED, 0, L"No models found");
        AppendMenuW(m, MF_STRING, ID_MODEL_BROWSE, L"Browse...");
    }
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);

    // Audio devices
    if (g_n_dev_in > 0) {
        for (int i = 0; i < g_n_dev_in && i < 4; i++)
            AppendMenuW(m, MF_STRING | (i == g_sel_in ? MF_CHECKED : 0),
                        ID_DEV_IN_BASE + i, g_dev_in[i].friendly);
    }
    if (g_n_dev_out > 0) {
        for (int i = 0; i < g_n_dev_out && i < 4; i++)
            AppendMenuW(m, MF_STRING | (i == g_sel_out ? MF_CHECKED : 0),
                        ID_DEV_OUT_BASE + i, g_dev_out[i].friendly);
    }
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);

    // Start / Stop
    AppendMenuW(m, MF_STRING | (g_running ? MF_GRAYED : 0), ID_START, L"\u25B6 Start");
    AppendMenuW(m, MF_STRING | (!g_running ? MF_GRAYED : 0), ID_STOP, L"\u25A0 Stop");
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);

    // Convert + Quit
    AppendMenuW(m, MF_STRING, ID_CONVERT, L"Convert file...");
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    AppendMenuW(m, MF_STRING, ID_QUIT, L"Quit");

    SetForegroundWindow(g_hwnd);
    int cmd = TrackPopupMenu(m, TPM_RIGHTBUTTON | TPM_RETURNCMD, pt.x, pt.y, 0, g_hwnd, NULL);
    DestroyMenu(m);

    // Handle selection
    if (cmd >= ID_BASE && cmd < ID_BASE + MAX_BACKENDS) {
        g_sel_be = cmd - ID_BASE;
        update_tip();
        save_config();
    } else if (cmd >= ID_MODEL_BASE && cmd < ID_MODEL_BASE + MAX_MODELS) {
        g_sel_model = cmd - ID_MODEL_BASE;
        save_config();
    } else if (cmd == ID_MODEL_BROWSE) {
        // Simple input dialog for model directory
        wchar_t buf[260];
        wcscpy_s(buf, 260, g_model_dir);
        // Use a basic approach: just rescan current dir for now
        // Full browse dialog needs comctl32 v6 manifest
        scan_models();
    } else if (cmd >= ID_DEV_IN_BASE && cmd < ID_DEV_IN_BASE + MAX_DEVS) {
        g_sel_in = cmd - ID_DEV_IN_BASE;
        save_config();
    } else if (cmd >= ID_DEV_OUT_BASE && cmd < ID_DEV_OUT_BASE + MAX_DEVS) {
        g_sel_out = cmd - ID_DEV_OUT_BASE;
        save_config();
    } else if (cmd == ID_START) {
        if (!g_running) {
            if (g_n_models == 0) {
                MessageBoxW(g_hwnd, L"No model selected.", L"Spartacv\u03C2", MB_ICONWARNING);
                return;
            }
            g_running = 1;
            update_tip();
            const wchar_t *in_id = (g_n_dev_in > 0) ? g_dev_in[g_sel_in].id : NULL;
            const wchar_t *out_id = (g_n_dev_out > 0) ? g_dev_out[g_sel_out].id : NULL;

            // Build full paths
            wchar_t model_path[300];
            swprintf(model_path, 300, L"%s\\%s", g_model_dir, g_models[g_sel_model]);

            wchar_t exe_dir[260];
            GetModuleFileNameW(NULL, exe_dir, 260);
            wchar_t *sl = wcsrchr(exe_dir, L'\\');
            if (sl) *(sl+1) = 0;
            wchar_t hubert[300], rmvpe[300];
            swprintf(hubert, 300, L"%smodels\base\vec-768-layer-12.onnx", exe_dir);
            swprintf(rmvpe, 300, L"%smodels\base\rmvpe.onnx", exe_dir);

            spcv_rt_start(model_path, g_backends[g_sel_be].be_id, 0,
                          in_id, out_id, hubert, rmvpe);
        }
    } else if (cmd == ID_STOP) {
        if (g_running) {
            spcv_rt_stop(NULL);
            g_running = 0;
            update_tip();
        }
    } else if (cmd == ID_CONVERT) {
        do_convert_file();
    } else if (cmd == ID_QUIT) {
        if (g_running) spcv_rt_stop(NULL);
        PostMessageW(g_hwnd, WM_CLOSE, 0, 0);
    }
}

// --- Window proc ---
static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_APP + 1:
        if (lp == WM_RBUTTONUP || lp == WM_LBUTTONUP) show_menu();
        return 0;
    case WM_CLOSE:
        Shell_NotifyIconW(NIM_DELETE, &g_nid);
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// --- Config persistence (simple key=value text file) ---
static wchar_t g_cfg_path[300] = L"";

static void cfg_path_init(void) {
    wchar_t exe_dir[260];
    GetModuleFileNameW(NULL, exe_dir, 260);
    wchar_t *sl = wcsrchr(exe_dir, L'\\');
    if (sl) *(sl+1) = 0;
    swprintf(g_cfg_path, 300, L"%sspartacvc.cfg", exe_dir);
}

static void save_config(void) {
    FILE *f = _wfopen(g_cfg_path, L"w");
    if (!f) return;
    fwprintf(f, L"backend=%d\n", g_sel_be);
    fwprintf(f, L"model=%s\n", g_n_models > 0 ? g_models[g_sel_model] : L"");
    fwprintf(f, L"model_dir=%s\n", g_model_dir);
    fwprintf(f, L"dev_in=%d\n", g_sel_in);
    fwprintf(f, L"dev_out=%d\n", g_sel_out);
    fclose(f);
}

static void load_config(void) {
    FILE *f = _wfopen(g_cfg_path, L"r");
    if (!f) return;
    wchar_t line[512];
    while (fgetws(line, 512, f)) {
        // strip newline
        int len = (int)wcslen(line);
        while (len > 0 && (line[len-1] == L'\n' || line[len-1] == L'\r')) line[--len] = 0;
        if (wcsncmp(line, L"backend=", 8) == 0) {
            int v = wcstol(line + 8, NULL, 10);
            if (v >= 0 && v < g_n_backends) g_sel_be = v;
        } else if (wcsncmp(line, L"model=", 6) == 0) {
            const wchar_t *name = line + 6;
            if (name[0]) {
                for (int i = 0; i < g_n_models; i++) {
                    if (wcscmp(g_models[i], name) == 0) { g_sel_model = i; break; }
                }
            }
        } else if (wcsncmp(line, L"model_dir=", 10) == 0) {
            const wchar_t *d = line + 10;
            if (d[0]) wcscpy_s(g_model_dir, 260, d);
        } else if (wcsncmp(line, L"dev_in=", 7) == 0) {
            int v = wcstol(line + 7, NULL, 10);
            if (v >= 0 && v < g_n_dev_in) g_sel_in = v;
        } else if (wcsncmp(line, L"dev_out=", 8) == 0) {
            int v = wcstol(line + 8, NULL, 10);
            if (v >= 0 && v < g_n_dev_out) g_sel_out = v;
        }
    }
    fclose(f);
}

// --- Main ---
int spcv_tray_run(void) {
    cfg_path_init();
    probe_backends();
    scan_models();
    scan_devices();
    load_config();  // restore saved selections

    WNDCLASSEXW wc = {0};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"SpartacvcTray";
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(0, L"SpartacvcTray", L"Spartacv\u03C2",
                             0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);
    if (!g_hwnd) return 1;

    // Try to load app icon from resource, fallback to system
    g_hicon = LoadIconW(GetModuleHandleW(NULL), MAKEINTRESOURCEW(1));
    if (!g_hicon) g_hicon = LoadIconW(NULL, IDI_APPLICATION);

    add_tray_icon();

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (g_hicon) DestroyIcon(g_hicon);
    wasapi_co_uninit();
    return 0;
}

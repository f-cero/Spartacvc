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
#include "index_discover.h"
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
#define ID_MODEL_CONVERT 391   // Convert models... dialog
#define ID_MODEL_RESIDENT 392  // toggle resident in VRAM
#define ID_MODEL_UNLOAD   393  // unload model
#define ID_DEV_IN_BASE   400   // input device: ID_DEV_IN_BASE + i
#define ID_DEV_OUT_BASE  500   // output device: ID_DEV_OUT_BASE + i
#define ID_DEV_INSTALL   590   // install virtual device
#define ID_DEV_UNINSTALL 591   // uninstall virtual device
#define ID_DEV_RENAME    592   // rename devices...
#define ID_PITCH_BASE    200   // pitch: ID_PITCH_BASE + (semitones + 12), range -12..+12
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
static int  g_pitch    = 0;   // semitones, -12..+12

// Model load state (drives tray icon)
typedef enum { MODEL_UNLOADED, MODEL_LOADING, MODEL_LOADED } model_state_t;
static model_state_t g_model_state = MODEL_UNLOADED;
static int  g_resident = 1;    // 1=keep in VRAM, 0=on-demand

static void save_config(void);

static NOTIFYICONDATAW g_nid = {0};
static HWND   g_hwnd  = NULL;
static HICON  g_hicon_loaded   = NULL;
static HICON  g_hicon_loading  = NULL;
static HICON  g_hicon_unloaded = NULL;

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
    wchar_t idx_path[520];
    wchar_t idx_arg[600] = L"";
    if (spcv_index_autodiscover(g_model_dir, g_models[g_sel_model], idx_path, 520))
        swprintf(idx_arg, 600, L" --index \"%s\"", idx_path);
    swprintf(cmd, 2048,
        L"\"%s\" \"%s\" --voice \"%s\" --hubert \"%s\" --rmvpe \"%s\" --backend %s --pitch %d%s --out \"%s\"",
        cli_path, in_file, model_path, hubert, rmvpe, be, g_pitch, idx_arg, out_file);

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
    // Icon set by current model state (default: unloaded)
    g_nid.hIcon = g_hicon_unloaded ? g_hicon_unloaded : g_hicon_loaded;
    wcscpy_s(g_nid.szTip, 128, L"Spartacv\u03C2");
    Shell_NotifyIconW(NIM_ADD, &g_nid);
}

// --- Model state → tray icon ---
static void update_tip(void);
static void set_model_state(model_state_t st) {
    g_model_state = st;
    HICON ic = NULL;
    switch (st) {
    case MODEL_LOADED:   ic = g_hicon_loaded;   break;
    case MODEL_LOADING:  ic = g_hicon_loading;  break;
    case MODEL_UNLOADED: ic = g_hicon_unloaded; break;
    }
    if (ic) {
        g_nid.hIcon = ic;
        Shell_NotifyIconW(NIM_MODIFY, &g_nid);
    }
    update_tip();
}

static void update_tip(void) {
    wchar_t tip[128];
    const wchar_t *model_name = (g_n_models > 0) ? g_models[g_sel_model] : L"none";
    const wchar_t *slash = wcsrchr(model_name, L'\\');
    if (slash) model_name = slash + 1;
    wchar_t mname[64];
    wcscpy_s(mname, 64, model_name);
    wchar_t *dot = wcsrchr(mname, L'.');
    if (dot) *dot = 0;

    const wchar_t *state = (g_model_state == MODEL_LOADED)   ? L""
                       : (g_model_state == MODEL_LOADING)  ? L" (loading\u2026)"
                       : L" (no model)";
    if (g_pitch != 0)
        swprintf(tip, 128, L"Spartacv\u03C2 [%s] %s%s %+d%s",
                 g_backends[g_sel_be].name, mname, state, g_pitch,
                 g_running ? L" \u25CF" : L"");
    else
        swprintf(tip, 128, L"Spartacv\u03C2 [%s] %s%s%s",
                 g_backends[g_sel_be].name, mname, state,
                 g_running ? L" \u25CF" : L"");
    wcscpy_s(g_nid.szTip, 128, tip);
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

// --- Context menu ---
static void show_menu(void) {
    HMENU m = CreatePopupMenu();
    POINT pt; GetCursorPos(&pt);

    // ---- Model submenu ----
    {
        HMENU sub = CreatePopupMenu();
        if (g_n_models > 0) {
            for (int i = 0; i < g_n_models; i++)
                AppendMenuW(sub, MF_STRING | (i == g_sel_model ? MF_CHECKED : 0),
                            ID_MODEL_BASE + i, g_models[i]);
        } else {
            AppendMenuW(sub, MF_GRAYED, 0, L"No models found");
        }
        AppendMenuW(sub, MF_SEPARATOR, 0, NULL);
        AppendMenuW(sub, MF_STRING, ID_MODEL_CONVERT,  L"Convert models\u2026");
        AppendMenuW(sub, MF_STRING | (g_resident ? MF_CHECKED : 0),
                    ID_MODEL_RESIDENT, L"Resident in VRAM");
        AppendMenuW(sub, MF_STRING | (g_model_state == MODEL_UNLOADED ? MF_GRAYED : 0),
                    ID_MODEL_UNLOAD,   L"Unload model");
        wchar_t model_title[80];
        const wchar_t *mn = (g_n_models > 0) ? g_models[g_sel_model] : L"none";
        const wchar_t *ms = wcsrchr(mn, L'\\'); if (ms) mn = ms + 1;
        wchar_t mbase[64]; wcscpy_s(mbase, 64, mn);
        wchar_t *mdot = wcsrchr(mbase, L'.'); if (mdot) *mdot = 0;
        swprintf(model_title, 80, L"Model: %s", mbase);
        AppendMenuW(m, MF_POPUP, (UINT_PTR)sub, model_title);
        AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    }

    // ---- Pitch submenu ----
    {
        HMENU sub = CreatePopupMenu();
        for (int s = -12; s <= 12; s++) {
            wchar_t lbl[24];
            if (s == 0)      swprintf(lbl, 24, L"0");
            else if (s > 0)  swprintf(lbl, 24, L"+%d", s);
            else             swprintf(lbl, 24, L"%d", s);
            AppendMenuW(sub, MF_STRING | (s == g_pitch ? MF_CHECKED : 0),
                        ID_PITCH_BASE + (s + 12), lbl);
        }
        wchar_t pitch_title[32];
        if (g_pitch == 0)      swprintf(pitch_title, 32, L"Pitch: 0");
        else if (g_pitch > 0)  swprintf(pitch_title, 32, L"Pitch: +%d", g_pitch);
        else                   swprintf(pitch_title, 32, L"Pitch: %d", g_pitch);
        AppendMenuW(m, MF_POPUP, (UINT_PTR)sub, pitch_title);
        AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    }

    // ---- Devices submenu ----
    {
        HMENU sub = CreatePopupMenu();
        AppendMenuW(sub, MF_GRAYED, 0, L"Input:");
        if (g_n_dev_in > 0) {
            for (int i = 0; i < g_n_dev_in && i < 6; i++)
                AppendMenuW(sub, MF_STRING | (i == g_sel_in ? MF_CHECKED : 0),
                            ID_DEV_IN_BASE + i, g_dev_in[i].friendly);
        } else AppendMenuW(sub, MF_GRAYED, 0, L"(none)");
        AppendMenuW(sub, MF_SEPARATOR, 0, NULL);
        AppendMenuW(sub, MF_GRAYED, 0, L"Output:");
        if (g_n_dev_out > 0) {
            for (int i = 0; i < g_n_dev_out && i < 6; i++)
                AppendMenuW(sub, MF_STRING | (i == g_sel_out ? MF_CHECKED : 0),
                            ID_DEV_OUT_BASE + i, g_dev_out[i].friendly);
        } else AppendMenuW(sub, MF_GRAYED, 0, L"(none)");
        AppendMenuW(sub, MF_SEPARATOR, 0, NULL);
        AppendMenuW(sub, MF_STRING, ID_DEV_INSTALL,   L"Install virtual device");
        AppendMenuW(sub, MF_STRING, ID_DEV_UNINSTALL, L"Uninstall virtual device");
        AppendMenuW(sub, MF_STRING, ID_DEV_RENAME,    L"Rename devices\u2026");
        AppendMenuW(m, MF_POPUP, (UINT_PTR)sub, L"Devices");
        AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    }

    // ---- Runtime submenu ----
    {
        HMENU sub = CreatePopupMenu();
        for (int i = 0; i < g_n_backends; i++) {
            if (!g_backends[i].available) continue;
            wchar_t lbl[256];
            swprintf(lbl, 256, L"%s \u2014 %s", g_backends[i].name, g_backends[i].desc);
            AppendMenuW(sub, MF_STRING | (i == g_sel_be ? MF_CHECKED : 0),
                        ID_BASE + i, lbl);
        }
        AppendMenuW(m, MF_POPUP, (UINT_PTR)sub, L"Runtime");
        AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    }

    // ---- Start / Stop ----
    AppendMenuW(m, MF_STRING | (g_running ? MF_GRAYED : 0), ID_START, L"\u25B6 Start");
    AppendMenuW(m, MF_STRING | (!g_running ? MF_GRAYED : 0), ID_STOP,  L"\u25A0 Stop");
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);

    // ---- Convert + Quit ----
    AppendMenuW(m, MF_STRING, ID_CONVERT, L"Convert file\u2026");
    AppendMenuW(m, MF_SEPARATOR, 0, NULL);
    AppendMenuW(m, MF_STRING, ID_QUIT, L"Quit");

    SetForegroundWindow(g_hwnd);
    int cmd = TrackPopupMenu(m, TPM_RIGHTBUTTON | TPM_RETURNCMD, pt.x, pt.y, 0, g_hwnd, NULL);
    // Destroy submenus (they're owned by m, but be safe)
    DestroyMenu(m);

    // Handle selection
    if (cmd >= ID_BASE && cmd < ID_BASE + MAX_BACKENDS) {
        g_sel_be = cmd - ID_BASE;
        update_tip();
        save_config();
    } else if (cmd >= ID_MODEL_BASE && cmd < ID_MODEL_BASE + MAX_MODELS) {
        g_sel_model = cmd - ID_MODEL_BASE;
        // Model selected → transition icon: loading → loaded.
        // (Real load happens in the model-handle refactor; for now we
        //  reflect the state so the tray icon is meaningful.)
        set_model_state(MODEL_LOADING);
        SetTimer(g_hwnd, 1, 400, NULL);   // flip to LOADED shortly
        save_config();
    } else if (cmd == ID_MODEL_RESIDENT) {
        g_resident = !g_resident;
        save_config();
    } else if (cmd == ID_MODEL_UNLOAD) {
        set_model_state(MODEL_UNLOADED);
        save_config();
    } else if (cmd == ID_MODEL_CONVERT) {
        // Conversion dialog — stub for now (full dialog is a later task).
        MessageBoxW(g_hwnd,
            L"Model conversion dialog is in progress.\n\n" L".pth \u2192 .gguf conversion will run here with a progress tracker.",
            L"Spartacv\u03C2 \u2014 Convert Models", MB_ICONINFORMATION);
    } else if (cmd == ID_DEV_INSTALL) {
        MessageBoxW(g_hwnd,
            L"Installing the Spartacv\u03C2 virtual audio device requires the\n" L"signed WASAPI driver + installer (see docs/tray-polish-spec.md \u00a75).\n\n" L"This is the 'just works' virtual mic — coming with the installer.",
            L"Spartacv\u03C2 \u2014 Virtual Device", MB_ICONINFORMATION);
    } else if (cmd == ID_DEV_UNINSTALL) {
        MessageBoxW(g_hwnd, L"No Spartacv\u03C2 virtual device is installed.",
            L"Spartacv\u03C2", MB_ICONINFORMATION);
    } else if (cmd == ID_DEV_RENAME) {
        MessageBoxW(g_hwnd,
            L"Device renaming will use the Windows device property store\n" L"(DEVPKEY_Device_FriendlyName). Available with the installer.\n\n" L"You can rename devices today in: Settings \u2192 System \u2192 Sound.",
            L"Spartacv\u03C2 \u2014 Rename Devices", MB_ICONINFORMATION);
    } else if (cmd == ID_MODEL_BROWSE) {
        // Simple input dialog for model directory
        wchar_t buf[260];
        wcscpy_s(buf, 260, g_model_dir);
        // Use a basic approach: just rescan current dir for now
        // Full browse dialog needs comctl32 v6 manifest
        scan_models();
    } else if (cmd >= ID_PITCH_BASE && cmd < ID_PITCH_BASE + 25) {
        g_pitch = cmd - ID_PITCH_BASE - 12;
        if (g_pitch < -12) g_pitch = -12;
        if (g_pitch > 12) g_pitch = 12;
        save_config();
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
                          in_id, out_id, hubert, rmvpe, g_pitch);
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
    case WM_TIMER:
        if (wp == 1) {   // model load finished (simulated)
            KillTimer(hwnd, 1);
            if (g_model_state == MODEL_LOADING)
                set_model_state(MODEL_LOADED);
        }
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
    fwprintf(f, L"pitch=%d\n", g_pitch);
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
        } else if (wcsncmp(line, L"pitch=", 6) == 0) {
            int v = wcstol(line + 6, NULL, 10);
            if (v >= -12 && v <= 12) g_pitch = v;
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

    HINSTANCE hinst = GetModuleHandleW(NULL);

    // Load the three state icons from the resource DLL.
    // IDs must match spartacvc.rc: 101=loaded 102=loading 103=unloaded
    g_hicon_loaded   = LoadIconW(hinst, MAKEINTRESOURCEW(101));
    g_hicon_loading  = LoadIconW(hinst, MAKEINTRESOURCEW(102));
    g_hicon_unloaded = LoadIconW(hinst, MAKEINTRESOURCEW(103));
    // Fallbacks if any resource is missing
    if (!g_hicon_loaded)   g_hicon_loaded   = LoadIconW(NULL, IDI_APPLICATION);
    if (!g_hicon_loading)  g_hicon_loading  = g_hicon_loaded;
    if (!g_hicon_unloaded) g_hicon_unloaded = g_hicon_loaded;

    WNDCLASSEXW wc = {0};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hinst;
    wc.hIcon = g_hicon_loaded;
    wc.hIconSm = g_hicon_loaded;
    wc.lpszClassName = L"SpartacvcTray";
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(0, L"SpartacvcTray", L"Spartacv\u03C2",
                             0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);
    if (!g_hwnd) return 1;

    // Initial model state: unloaded (icon reflects this)
    g_model_state = MODEL_UNLOADED;
    add_tray_icon();

    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    if (g_hicon_loaded)   DestroyIcon(g_hicon_loaded);
    if (g_hicon_loading)  DestroyIcon(g_hicon_loading);
    if (g_hicon_unloaded) DestroyIcon(g_hicon_unloaded);
    wasapi_co_uninit();
    return 0;
}

/**
 * @file    gpu_list.c
 * @brief   Enumerate DirectML-capable GPUs for `--gpu list`.
 *
 * For each DXGI adapter, tries D3D12CreateDevice + DMLCreateDevice.
 * Adapters where both succeed are DirectML-capable; their index is the
 * value to pass to --gpu (the DML provider's device_id).
 *
 * Pure C: COM methods via lpVtbl (CINTERFACE), IIDs hardcoded from SDK
 * headers (initguid.h is unreliable in C). DirectML.h is C++-incompatible,
 * so DMLCreateDevice is loaded dynamically from directml.dll (C-ABI export).
 */
#define WIN32_LEAN_AND_MEAN
#define CINTERFACE
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
#include <dxgi1_2.h>    /* IDXGIFactory1, EnumAdapters1, DXGI_ADAPTER_DESC1 */
#include <d3d12.h>       /* D3D12CreateDevice, ID3D12Device */

/* --- IIDs (hardcoded from SDK headers; *_sp suffix avoids EXTERN_C clash) --- */
/* dxgi.h:  DEFINE_GUID(IID_IDXGIFactory1, 0x770aae78,0xf26f,0x4dba, ...) */
static const IID IID_IDXGIFactory1_sp =
    {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
/* d3d12.h: DEFINE_GUID(IID_ID3D12Device, 0x189819f1,0x1db6,0x4b57, ...) */
static const IID IID_ID3D12Device_sp =
    {0x189819f1, 0x1db6, 0x4b57, {0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7}};

/* --- DirectML (dynamic load, avoids C++-only DirectML.h) --- */
typedef HRESULT (WINAPI *PFN_DMLCreateDevice)(
    void *d3d12Device, unsigned int flags, const void *riid, void **ppv);
/* DirectML.h: DML_DECLARE_INTERFACE("6dbd6437-96fd-423f-a98c-ae5e7c2a573f") */
static const IID IID_IDMLDevice_sp =
    {0x6dbd6437, 0x96fd, 0x423f, {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}};

static PFN_DMLCreateDevice g_DMLCreateDevice = NULL;
static HMODULE g_dml_mod = NULL;

static int dml_load(void) {
    if (g_DMLCreateDevice) return 1;
    g_dml_mod = LoadLibraryW(L"directml.dll");
    if (!g_dml_mod) return 0;
    g_DMLCreateDevice = (PFN_DMLCreateDevice)(void *)
        GetProcAddress(g_dml_mod, "DMLCreateDevice");
    return g_DMLCreateDevice != NULL;
}

static void print_wide(const WCHAR *text) {
    if (!text) { printf("(null)"); return; }
    int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, NULL, 0, NULL, NULL);
    if (size <= 0) { printf("?"); return; }
    char *buf = (char *)malloc((size_t)size);
    if (!buf) { printf("?"); return; }
    WideCharToMultiByte(CP_UTF8, 0, text, -1, buf, size, NULL, NULL);
    printf("%s", buf);
    free(buf);
}

static const char *vendor_name(UINT vid) {
    if (vid == 0x10de) return "NVIDIA";
    if (vid == 0x1002) return "AMD/ATI";
    if (vid == 0x8086) return "Intel";
    if (vid == 0x1414) return "Microsoft";
    return "other";
}

/* Returns 1 if the adapter is DirectML-capable (D3D12 + DML device create OK). */
static int probe_directml(IDXGIAdapter1 *adapter) {
    ID3D12Device *d3d12 = NULL;
    HRESULT hr = D3D12CreateDevice((IUnknown *)adapter, D3D_FEATURE_LEVEL_11_0,
                                   &IID_ID3D12Device_sp, (void **)&d3d12);
    if (FAILED(hr) || !d3d12) return 0;
    int ok = 0;
    if (g_DMLCreateDevice) {
        void *dml = NULL;
        hr = g_DMLCreateDevice(d3d12, 0, &IID_IDMLDevice_sp, &dml);
        ok = SUCCEEDED(hr) && dml;
        /* DML device is not released here: without DirectML.h we lack the
         * proper interface to call Release. The D3D12 device release below
         * and process exit handle cleanup. Acceptable for a one-shot probe. */
    }
    d3d12->lpVtbl->Release(d3d12);
    return ok;
}

int spcv_list_gpus(void) {
    if (!dml_load()) {
        fprintf(stderr, "directml.dll not found (DirectML runtime missing)\n");
        return -1;
    }
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        fprintf(stderr, "CoInitializeEx failed: 0x%08lx\n", (unsigned long)hr);
        return -1;
    }
    IDXGIFactory1 *factory = NULL;
    hr = CreateDXGIFactory1(&IID_IDXGIFactory1_sp, (void **)&factory);
    if (FAILED(hr) || !factory) {
        fprintf(stderr, "CreateDXGIFactory1 failed: 0x%08lx\n", (unsigned long)hr);
        CoUninitialize();
        return -1;
    }
    int dml_count = 0;
    for (UINT i = 0; ; i++) {
        IDXGIAdapter1 *adapter = NULL;
        hr = factory->lpVtbl->EnumAdapters1(factory, i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr) || !adapter) break;
        DXGI_ADAPTER_DESC1 desc;
        int have_desc = SUCCEEDED(adapter->lpVtbl->GetDesc1(adapter, &desc));
        int dml_ok = have_desc && probe_directml(adapter);
        if (have_desc) {
            printf("GPU %u: ", (unsigned)i);
            print_wide(desc.Description);
            printf("   [%s, vendor=0x%04x, vram=%.1f GB, %s]\n",
                   vendor_name(desc.VendorId), (unsigned)desc.VendorId,
                   (double)desc.DedicatedVideoMemory / (1024.0 * 1024.0 * 1024.0),
                   (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ? "software" : "hardware");
            printf("       DirectML: %s   ->  --gpu %u\n",
                   dml_ok ? "YES" : "no", (unsigned)i);
        }
        if (dml_ok) dml_count++;
        adapter->lpVtbl->Release(adapter);
    }
    factory->lpVtbl->Release(factory);
    CoUninitialize();
    if (g_dml_mod) FreeLibrary(g_dml_mod);
    if (dml_count == 0) printf("\n(No DirectML-capable devices found)\n");
    else printf("\n%d DirectML-capable device(s). Use --gpu <index> to select.\n", dml_count);
    return dml_count;
}

#else
int spcv_list_gpus(void) {
    fprintf(stderr, "--gpu list is only supported on Windows (DXGI/D3D12/DML)\n");
    return -1;
}
#endif

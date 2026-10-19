/**
 * @file    gpu_list.c
 * @brief   Enumerate DXGI GPU adapters for `--gpu list`.
 *
 * Prints each adapter's index (the value to pass to --gpu) and description.
 * The DML provider's `device_id` option is the DXGI adapter index, so the
 * index printed here is exactly what --gpu <id> expects.
 *
 * Compiled as C, so COM methods are called through the vtable (lpVtbl).
 */
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <dxgi.h>

/* IID for IDXGIFactory (from dxgi.h: MIDL_INTERFACE("7b7166ec-21c7-44ae-b21a-c9ae321ae369")) */
static const IID IID_IDXGIFactory =
    {0x7b7166ec, 0x21c7, 0x44ae, {0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69}};

int spcv_list_gpus(void) {
    IDXGIFactory *factory = NULL;
    HRESULT hr = CreateDXGIFactory(&IID_IDXGIFactory, (void **)&factory);
    if (FAILED(hr) || !factory) {
        fprintf(stderr, "CreateDXGIFactory failed: 0x%08lx\n", (unsigned long)hr);
        return -1;
    }
    int count = 0;
    for (UINT i = 0; ; i++) {
        IDXGIAdapter *adapter = NULL;
        hr = factory->lpVtbl->EnumAdapters(factory, i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(hr) || !adapter) break;
        DXGI_ADAPTER_DESC desc;
        if (SUCCEEDED(adapter->lpVtbl->GetDesc(adapter, &desc))) {
            const char *vend =
                (desc.VendorId == 0x10de) ? "NVIDIA" :
                (desc.VendorId == 0x1002) ? "AMD/ATI" :
                (desc.VendorId == 0x8086) ? "Intel" : "other";
            printf("GPU %u: %ls   [%s, vendor=0x%04x device=0x%04x]\n",
                   (unsigned)i, desc.Description, vend,
                   (unsigned)desc.VendorId, (unsigned)desc.DeviceId);
            count++;
        }
        adapter->lpVtbl->Release(adapter);
    }
    factory->lpVtbl->Release(factory);
    if (count == 0) printf("(no DXGI adapters found)\n");
    else printf("\nUse: --gpu <index>   (e.g. --gpu 0)\n");
    return count;
}

#else
int spcv_list_gpus(void) {
    fprintf(stderr, "--gpu list is only supported on Windows (DXGI)\n");
    return -1;
}
#endif

// wasapi.c — WASAPI capture/render implementation.
//
// Shared-mode, 32-bit float. COM in C (no C++ smart pointers).
// Capture: GetBuffer → memcpy → ReleaseBuffer (polling loop).
// Render:  GetBuffer → memcpy → ReleaseBuffer (polling loop).

#define INITGUID  // Must be before guiddef.h (pulled in by windows.h)
#define COBJMACROS
#include "wasapi.h"
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <AudioSessionTypes.h>
#include <devpkey.h>
#include <propsys.h>
#include <mmreg.h>
#include <stdio.h>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

int wasapi_co_init(void) {
    return (int)CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
}

void wasapi_co_uninit(void) {
    CoUninitialize();
}

// HRESULTs not in older SDK headers
#ifndef AUDCLNT_E_BUFFER_EMPTY
#define AUDCLNT_E_BUFFER_EMPTY  ((HRESULT)0x88890004)
#endif
#ifndef AUDCLNT_E_BUFFER_FULL
#define AUDCLNT_E_BUFFER_FULL   ((HRESULT)0x8889000A)
#endif

// --- Helper: get a device ---
static IMMDevice * get_device(const wchar_t *dev_id, EDataFlow flow) {
    IMMDeviceEnumerator *enumr = NULL;
    IMMDevice *dev = NULL;
    HRESULT hr;

    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &IID_IMMDeviceEnumerator, (void **)&enumr);
    if (FAILED(hr)) return NULL;

    if (dev_id)
        hr = IMMDeviceEnumerator_GetDevice(enumr, dev_id, &dev);
    else
        hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(enumr, flow, eConsole, &dev);

    IMMDeviceEnumerator_Release(enumr);
    return SUCCEEDED(hr) ? dev : NULL;
}

// --- Helper: activate IAudioClient ---
static IAudioClient * activate_client(IMMDevice *dev) {
    IAudioClient *client = NULL;
    HRESULT hr = IMMDevice_Activate(dev, &IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&client);
    IMMDevice_Release(dev);
    return SUCCEEDED(hr) ? client : NULL;
}

// --- Helper: get 32-bit float format ---
static WAVEFORMATEX * get_float_format(IAudioClient *client, int *rate, int *chans) {
    WAVEFORMATEX *mix = NULL;
    IAudioClient_GetMixFormat(client, &mix);
    if (!mix) return NULL;

    if (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        // WAVEFORMATEXTENSIBLE: first 18 bytes are WAVEFORMATEX, then extras
        // We just report the base format; WASAPI will auto-convert to float32
        *rate = mix->nSamplesPerSec;
        *chans = mix->nChannels;
        return mix;
    }
    *rate = mix->nSamplesPerSec;
    *chans = mix->nChannels;
    return mix;
}

// ==================== CAPTURE ====================

struct wasapi_cap {
    IAudioClient *client;
    IAudioCaptureClient *cap;
    int rate;
    int channels;
};

wasapi_cap_t * wasapi_capture_open(const wchar_t *dev_id) {
    IMMDevice *dev = get_device(dev_id, eCapture);
    if (!dev) return NULL;

    IAudioClient *client = activate_client(dev);
    if (!client) return NULL;

    int rate, chans;
    WAVEFORMATEX *fmt = get_float_format(client, &rate, &chans);
    if (!fmt) { IAudioClient_Release(client); return NULL; }

    HRESULT hr = IAudioClient_Initialize(client, AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
        200000, 0, fmt, NULL);
    if (FAILED(hr)) { IAudioClient_Release(client); return NULL; }

    IAudioCaptureClient *cap = NULL;
    hr = IAudioClient_GetService(client, &IID_IAudioCaptureClient, (void **)&cap);
    if (FAILED(hr)) { IAudioClient_Release(client); return NULL; }

    wasapi_cap_t *c = (wasapi_cap_t *)calloc(1, sizeof(*c));
    c->client = client;
    c->cap = cap;
    c->rate = rate;
    c->channels = chans;
    return c;
}

int wasapi_capture_read(wasapi_cap_t *c, float *buf, int n) {
    if (!c || !c->cap) return -1;

    BYTE *data = NULL;
    UINT32 frames = 0;
    DWORD pkt_size = 0, flags = 0;

    HRESULT hr;
    int tries = 0;
    while ((hr = IAudioCaptureClient_GetBuffer(c->cap, &data, &frames, &flags, &pkt_size, NULL))
           == AUDCLNT_E_BUFFER_EMPTY) {
        if (++tries > 100) return -1;
        Sleep(1);
    }
    if (FAILED(hr)) return -1;

    int max_s = (int)frames;
    if (max_s > n) max_s = n;
    memcpy(buf, data, (size_t)max_s * c->channels * sizeof(float));
    IAudioCaptureClient_ReleaseBuffer(c->cap, frames);
    return max_s;
}

int wasapi_capture_rate(const wasapi_cap_t *c) { return c ? c->rate : 0; }
int wasapi_capture_channels(const wasapi_cap_t *c) { return c ? c->channels : 0; }

void wasapi_capture_close(wasapi_cap_t *c) {
    if (!c) return;
    if (c->cap) IAudioCaptureClient_Release(c->cap);
    if (c->client) IAudioClient_Release(c->client);
    free(c);
}

// ==================== RENDER ====================

struct wasapi_render {
    IAudioClient *client;
    IAudioRenderClient *rnd;
    int rate;
    int channels;
};

wasapi_render_t * wasapi_render_open(const wchar_t *dev_id) {
    IMMDevice *dev = get_device(dev_id, eRender);
    if (!dev) return NULL;

    IAudioClient *client = activate_client(dev);
    if (!client) return NULL;

    int rate, chans;
    WAVEFORMATEX *fmt = get_float_format(client, &rate, &chans);
    if (!fmt) { IAudioClient_Release(client); return NULL; }

    HRESULT hr = IAudioClient_Initialize(client, AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
        200000, 0, fmt, NULL);
    if (FAILED(hr)) { IAudioClient_Release(client); return NULL; }

    IAudioRenderClient *rnd = NULL;
    hr = IAudioClient_GetService(client, &IID_IAudioRenderClient, (void **)&rnd);
    if (FAILED(hr)) { IAudioClient_Release(client); return NULL; }

    wasapi_render_t *r = (wasapi_render_t *)calloc(1, sizeof(*r));
    r->client = client;
    r->rnd = rnd;
    r->rate = rate;
    r->channels = chans;
    return r;
}

int wasapi_render_write(wasapi_render_t *r, const float *buf, int n) {
    if (!r || !r->rnd) return -1;

    int written = 0;
    while (written < n) {
        int chunk = n - written;
        BYTE *data = NULL;
        HRESULT hr = IAudioRenderClient_GetBuffer(r->rnd, (UINT32)chunk, &data);
        if (hr == AUDCLNT_E_BUFFER_FULL) { Sleep(1); continue; }
        if (FAILED(hr)) return -1;
        memcpy(data, buf + written, (size_t)chunk * r->channels * sizeof(float));
        IAudioRenderClient_ReleaseBuffer(r->rnd, (UINT32)chunk, 0);
        written += chunk;
    }
    return 0;
}

int wasapi_render_rate(const wasapi_render_t *r) { return r ? r->rate : 0; }
int wasapi_render_channels(const wasapi_render_t *r) { return r ? r->channels : 0; }

int wasapi_render_start(wasapi_render_t *r) {
    if (!r) return -1;
    return (int)IAudioClient_Start(r->client);
}

int wasapi_render_stop(wasapi_render_t *r) {
    if (!r) return -1;
    return (int)IAudioClient_Stop(r->client);
}

void wasapi_render_close(wasapi_render_t *r) {
    if (!r) return;
    if (r->rnd) IAudioRenderClient_Release(r->rnd);
    if (r->client) IAudioClient_Release(r->client);
    free(r);
}

// ==================== ENUMERATION ====================

static int enum_devices(EDataFlow flow, wasapi_dev_t *devs, int max_n) {
    IMMDeviceEnumerator *enumr = NULL;
    HRESULT hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                                  &IID_IMMDeviceEnumerator, (void **)&enumr);
    if (FAILED(hr)) return -1;

    IMMDeviceCollection *coll = NULL;
    hr = IMMDeviceEnumerator_EnumAudioEndpoints(enumr, flow, DEVICE_STATE_ACTIVE, &coll);
    if (FAILED(hr)) { IMMDeviceEnumerator_Release(enumr); return -1; }

    UINT count = 0;
    IMMDeviceCollection_GetCount(coll, &count);

    int n = 0;
    for (UINT i = 0; i < count && n < max_n; i++) {
        IMMDevice *dev = NULL;
        if (FAILED(IMMDeviceCollection_Item(coll, i, &dev))) continue;

        PWSTR id_str = NULL;
        IMMDevice_GetId(dev, &id_str);

        WCHAR friendly[256] = L"(unknown)";
        IPropertyStore *store = NULL;
        if (SUCCEEDED(dev->lpVtbl->OpenPropertyStore(dev, STGM_READ, (IUnknown **)&store))) {
            PROPVARIANT pv;
            ZeroMemory(&pv, sizeof(pv));
            if (SUCCEEDED(store->lpVtbl->GetValue(store, &DEVPKEY_Device_FriendlyName, &pv))) {
                if (pv.vt == VT_BSTR && pv.pwszVal)
                    lstrcpynW(friendly, pv.pwszVal, 256);
                PropVariantClear(&pv);
            }
            store->lpVtbl->Release(store);
        }

        if (id_str) {
            wcsncpy_s(devs[n].id, 256, id_str, 255);
            CoTaskMemFree(id_str);
        }
        wcsncpy_s(devs[n].friendly, 256, friendly, 255);
        devs[n].is_default_in = 0;
        devs[n].is_default_out = 0;
        n++;
        IMMDevice_Release(dev);
    }

    IMMDeviceCollection_Release(coll);
    IMMDeviceEnumerator_Release(enumr);
    return n;
}

int wasapi_enum_inputs(wasapi_dev_t *devs, int max_n) {
    return enum_devices(eCapture, devs, max_n);
}

int wasapi_enum_outputs(wasapi_dev_t *devs, int max_n) {
    return enum_devices(eRender, devs, max_n);
}

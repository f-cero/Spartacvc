// wasapi.c — WASAPI capture/render implementation.
//
// Uses COM + IAudioClient in shared mode with 32-bit float format.
// Capture: reads from mic in a blocking loop.
// Render:  writes to speakers/virtual cable in a blocking loop.
//
// COM interfaces are used directly (no C++ smart pointers).
// All COM pointers are freed on close.

#define COBJMACROS
#include "wasapi.h"
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <stdio.h>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

static HRESULT g_com_state = CO_E_NOTINITIALIZED;

int wasapi_co_init(void) {
    return CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
}

void wasapi_co_uninit(void) {
    if (g_com_state == S_OK) CoUninitialize();
}

// --- Helper: get a device ---
static IMMDevice * get_device(const wchar_t *dev_id, ERole role) {
    IMMDeviceEnumerator *enumr = NULL;
    IMMDevice *dev = NULL;
    HRESULT hr;

    hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                          &IID_IMMDeviceEnumerator, (void **)&enumr);
    if (FAILED(hr)) return NULL;

    if (dev_id)
        hr = enumr->lpVtbl->GetDevice(enumr, dev_id, &dev);
    else
        hr = enumr->lpVtbl->GetDefaultAudioEndpoint(enumr, role, eConsole, &dev);

    enumr->lpVtbl->Release(enumr);
    return (SUCCEEDED(hr)) ? dev : NULL;
}

// --- Helper: activate IAudioClient from a device ---
static IAudioClient * activate_client(IMMDevice *dev) {
    IAudioClient *client = NULL;
    HRESULT hr = dev->lpVtbl->Activate(dev, &IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&client);
    dev->lpVtbl->Release(dev);
    return SUCCEEDED(hr) ? client : NULL;
}

// --- Helper: negotiate 32-bit float mix format ---
static WAVEFORMATEX * get_float_format(IAudioClient *client, int *rate, int *chans) {
    WAVEFORMATEX *mix = NULL;
    client->lpVtbl->GetMixFormat(client, &mix);
    if (mix->wFormatTag == WAVEFORMATEXTENSIBLE) {
        WAVEFORMATEXTENSIBLE *ext = (WAVEFORMATEXTENSIBLE *)mix;
        // Convert to plain PCM float
        static WAVEFORMATEX pcm;
        ZeroMemory(&pcm, sizeof(pcm));
        pcm.wFormatTag = WAVEFORMATEX_PCM;
        pcm.nChannels = ext->nChannels;
        pcm.nSamplesPerSec = ext->nSamplesPerSec;
        pcm.nBlockAlign = ext->nChannels * 4;
        pcm.nAvgBytesPerSec = ext->nSamplesPerSec * ext->nChannels * 4;
        pcm.wBitsPerSample = 32;
        *rate = ext->nSamplesPerSec;
        *chans = ext->nChannels;
        return &pcm;
    }
    *rate = mix->nSamplesPerSec;
    *chans = mix->nChannels;
    return mix;
}

// ==================== CAPTURE ====================

struct wasapi_cap {
    IAudioClient    *client;
    IAudioCaptureClient *cap;
    WAVEFORMATEX    *fmt;
    int             rate;
    int             channels;
};

wasapi_cap_t * wasapi_capture_open(const wchar_t *dev_id) {
    IMMDevice *dev = get_device(dev_id, eCapture);
    if (!dev) return NULL;

    IAudioClient *client = activate_client(dev);
    if (!client) return NULL;

    int rate, chans;
    WAVEFORMATEX *fmt = get_float_format(client, &rate, &chans);

    // Initialize in shared mode, event-driven
    AUDCLNT_STREAMFLAGS flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    HRESULT hr = client->lpVtbl->Initialize(client, AUDCLNT_SHAREMODE_SHARED, flags,
                                             200000, 0, fmt, NULL); // 200ms buffer
    if (FAILED(hr)) { client->lpVtbl->Release(client); return NULL; }

    IAudioCaptureClient *cap = NULL;
    hr = client->lpVtbl->GetService(client, &IID_IAudioCaptureClient, (void **)&cap);
    if (FAILED(hr)) { client->lpVtbl->Release(client); return NULL; }

    wasapi_cap_t *c = (wasapi_cap_t *)calloc(1, sizeof(*c));
    c->client = client;
    c->cap = cap;
    c->fmt = fmt;
    c->rate = rate;
    c->channels = chans;
    return c;
}

int wasapi_capture_read(wasapi_cap_t *c, float *buf, int n) {
    if (!c || !c->cap) return -1;

    // Get available data
    UINT32 frames = 0;
    BYTE *data = NULL;
    DWORD pkt_size = 0;
    DWORD flags = 0;

    // Wait for data (blocking via GetBuffer in a loop)
    HRESULT hr;
    while ((hr = c->cap->lpVtbl->GetBuffer(c->cap, &data, &frames, &flags, &pkt_size, NULL)) == AUDCLNT_E_BUFFER_EMPTY) {
        Sleep(1);
    }
    if (FAILED(hr)) return -1;

    int max_samples = (int)frames;
    if (max_samples > n) max_samples = n;

    memcpy(buf, data, (size_t)max_samples * c->channels * sizeof(float));
    c->cap->lpVtbl->ReleaseBuffer(c->cap, frames);
    return max_samples;
}

int wasapi_capture_rate(const wasapi_cap_t *c) { return c ? c->rate : 0; }
int wasapi_capture_channels(const wasapi_cap_t *c) { return c ? c->channels : 0; }

void wasapi_capture_close(wasapi_cap_t *c) {
    if (!c) return;
    if (c->cap) c->cap->lpVtbl->Release(c->cap);
    if (c->client) c->client->lpVtbl->Release(c->client);
    free(c);
}

// ==================== RENDER ====================

struct wasapi_render {
    IAudioClient    *client;
    IAudioRenderClient *rnd;
    WAVEFORMATEX    *fmt;
    int             rate;
    int             channels;
};

wasapi_render_t * wasapi_render_open(const wchar_t *dev_id) {
    IMMDevice *dev = get_device(dev_id, eRender);
    if (!dev) return NULL;

    IAudioClient *client = activate_client(dev);
    if (!client) return NULL;

    int rate, chans;
    WAVEFORMATEX *fmt = get_float_format(client, &rate, &chans);

    AUDCLNT_STREAMFLAGS flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
    HRESULT hr = client->lpVtbl->Initialize(client, AUDCLNT_SHAREMODE_SHARED, flags,
                                             200000, 0, fmt, NULL);
    if (FAILED(hr)) { client->lpVtbl->Release(client); return NULL; }

    IAudioRenderClient *rnd = NULL;
    hr = client->lpVtbl->GetService(client, &IID_IAudioRenderClient, (void **)&rnd);
    if (FAILED(hr)) { client->lpVtbl->Release(client); return NULL; }

    wasapi_render_t *r = (wasapi_render_t *)calloc(1, sizeof(*r));
    r->client = client;
    r->rnd = rnd;
    r->fmt = fmt;
    r->rate = rate;
    r->channels = chans;
    return r;
}

int wasapi_render_write(wasapi_render_t *r, const float *buf, int n) {
    if (!r || !r->rnd) return -1;

    // Write in chunks if buffer is too small
    int written = 0;
    while (written < n) {
        int chunk = n - written;
        BYTE *data = NULL;
        HRESULT hr = r->rnd->lpVtbl->GetBuffer(r->rnd, chunk, &data);
        if (FAILED(hr)) {
            if (hr == AUDCLNT_E_BUFFER_FULL) { Sleep(1); continue; }
            return -1;
        }
        memcpy(data, buf + written, (size_t)chunk * r->channels * sizeof(float));
        r->rnd->lpVtbl->Write(r->rnd, chunk, 0, data);
        written += chunk;
    }
    return 0;
}

int wasapi_render_rate(const wasapi_render_t *r) { return r ? r->rate : 0; }
int wasapi_render_channels(const wasapi_render_t *r) { return r ? r->channels : 0; }

int wasapi_render_start(wasapi_render_t *r) {
    if (!r) return -1;
    return (int)r->client->lpVtbl->Start(r->client);
}

int wasapi_render_stop(wasapi_render_t *r) {
    if (!r) return -1;
    return (int)r->client->lpVtbl->Stop(r->client);
}

void wasapi_render_close(wasapi_render_t *r) {
    if (!r) return;
    if (r->rnd) r->rnd->lpVtbl->Release(r->rnd);
    if (r->client) r->client->lpVtbl->Release(r->client);
    free(r);
}

// ==================== ENUMERATION ====================

static int enum_devices(eDataFlow flow, wasapi_dev_t *devs, int max_n) {
    IMMDeviceEnumerator *enumr = NULL;
    HRESULT hr = CoCreateInstance(&CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL,
                                  &IID_IMMDeviceEnumerator, (void **)&enumr);
    if (FAILED(hr)) return -1;

    IMMDeviceCollection *coll = NULL;
    hr = enumr->lpVtbl->EnumAudioEndpoints(enumr, flow, DEVICE_STATE_ACTIVE, &coll);
    if (FAILED(hr)) { enumr->lpVtbl->Release(enumr); return -1; }

    UINT count = 0;
    coll->lpVtbl->GetCount(coll, &count);

    int n = 0;
    for (UINT i = 0; i < count && n < max_n; i++) {
        IMMDevice *dev = NULL;
        if (FAILED(coll->lpVtbl->Item(coll, i, &dev))) continue;

        PWSTR id_str = NULL;
        dev->lpVtbl->GetId(dev, &id_str);

        WCHAR friendly[256] = L"(unknown)";
        PROPVARIANT pv;
        ZeroMemory(&pv, sizeof(pv));
        if (SUCCEEDED(dev->lpVtbl->GetProperty(dev, &DEVPKEY_Device_FriendlyName, &pv))) {
            if (pv.vt == VT_BSTR && pv.val.bstrVal)
                wcsncpy_s(friendly, 256, pv.val.bstrVal, 255);
            PropVariantClear(&pv);
        }

        if (id_str) {
            wcsncpy_s(devs[n].id, 256, id_str, 255);
            CoTaskMemFree(id_str);
        }
        wcsncpy_s(devs[n].friendly, 256, friendly, 255);
        devs[n].is_default_in = 0;
        devs[n].is_default_out = 0;
        n++;

        dev->lpVtbl->Release(dev);
    }

    coll->lpVtbl->Release(coll);
    enumr->lpVtbl->Release(enumr);
    return n;
}

int wasapi_enum_inputs(wasapi_dev_t *devs, int max_n) {
    return enum_devices(eCapture, devs, max_n);
}

int wasapi_enum_outputs(wasapi_dev_t *devs, int max_n) {
    return enum_devices(eRender, devs, max_n);
}

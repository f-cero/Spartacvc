// pipeline_rt.c — Real-time RVC pipeline worker thread.
//
// Strategy: accumulate ~1s of audio, write temp WAV, call spartacvc.exe CLI,
// read output WAV, play back via WASAPI render.
// Latency: ~1-3s. Not true real-time, but produces actual RVC output.
//
// TODO: Link pipeline code directly for true real-time (no CLI round-trip).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wasapi.h"
#include "pipeline_rt.h"

static volatile int g_stop = 0;
static volatile int g_running = 0;
static HANDLE g_thread = NULL;

// Config passed from tray menu
static wchar_t g_model[300]   = L"";
static wchar_t g_hubert[300]  = L"";
static wchar_t g_rmvpe[300]   = L"";
static wchar_t g_in_dev[260]  = L"";
static wchar_t g_out_dev[260] = L"";
static wchar_t g_exe_dir[260] = L"";
static int     g_be_id        = 0;

// --- WAV I/O helpers (16-bit PCM mono) ---

static int wav_write_16(const wchar_t *path, const float *data, int64_t n, int sr) {
    FILE *f = _wfopen(path, L"wb");
    if (!f) return -1;
    uint32_t data_size = (uint32_t)(n * 2);
    uint32_t total = 36 + data_size;
    // RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&total, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t fmt = 1, ch = 1, bits = 16;
    uint32_t byte_rate = sr * 2;
    uint16_t block_align = 2;
    fwrite(&fmt, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    // data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    // samples (float → 16-bit PCM)
    for (int64_t i = 0; i < n; i++) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}

typedef struct { float *data; int64_t n; int sr; } wav_buf_t;

static int wav_read_f(const wchar_t *path, wav_buf_t *w) {
    FILE *f = _wfopen(path, L"rb");
    if (!f) return -1;
    char hdr[12];
    if (fread(hdr, 1, 12, f) != 12) { fclose(f); return -1; }
    if (memcmp(hdr, "RIFF", 4) || memcmp(hdr+8, "WAVE", 4)) { fclose(f); return -1; }
    int sr = 0; int found_data = 0;
    int64_t data_size = 0;
    while (!found_data && !feof(f)) {
        char cid[4]; uint32_t csize;
        if (fread(cid, 1, 4, f) != 4) break;
        if (fread(&csize, 4, 1, f) != 4) break;
        if (memcmp(cid, "fmt ", 4) == 0) {
            uint16_t fmt, ch; uint32_t tmp;
            fread(&fmt, 2, 1, f);
            fread(&ch, 2, 1, f);
            fread(&sr, 4, 1, f);
            fseek(f, csize - 8, SEEK_CUR);
        } else if (memcmp(cid, "data", 4) == 0) {
            data_size = csize;
            found_data = 1;
        } else {
            fseek(f, csize, SEEK_CUR);
        }
    }
    if (!found_data) { fclose(f); return -1; }
    int64_t n = data_size / 2; // 16-bit mono
    w->data = (float *)malloc((size_t)n * sizeof(float));
    if (!w->data) { fclose(f); return -1; }
    for (int64_t i = 0; i < n; i++) {
        int16_t s;
        if (fread(&s, 2, 1, f) != 1) { free(w->data); fclose(f); return -1; }
        w->data[i] = (float)s / 32768.0f;
    }
    w->n = n;
    w->sr = sr;
    fclose(f);
    return 0;
}

// --- Batch conversion worker ---

static DWORD WINAPI rt_worker(LPVOID arg) {
    (void)arg;
    g_running = 1;

    wasapi_co_init();
    const wchar_t *in_id  = g_in_dev[0]  ? g_in_dev  : NULL;
    const wchar_t *out_id = g_out_dev[0] ? g_out_dev : NULL;
    wasapi_cap_t *cap = wasapi_capture_open(in_id);
    wasapi_render_t *rnd = wasapi_render_open(out_id);

    if (!cap || !rnd) {
        if (cap) wasapi_capture_close(cap);
        if (rnd) wasapi_render_close(rnd);
        wasapi_co_uninit();
        g_running = 0;
        return 1;
    }

    int in_rate = wasapi_capture_rate(cap);
    int in_ch   = wasapi_capture_channels(cap);
    int out_rate = wasapi_render_rate(rnd);
    int out_ch   = wasapi_render_channels(rnd);

    wasapi_render_start(rnd);

    // Batch: 1 second of audio
    int batch_samples = in_rate; // 1s
    float *batch = (float *)malloc((size_t)batch_samples * in_ch * sizeof(float));
    int batch_count = 0;

    const wchar_t *in_path  = L"C:\\temp\\spcv_rt_in.wav";
    const wchar_t *out_path = L"C:\\temp\\spcv_rt_out.wav";

    const wchar_t *be_str = (g_be_id == 1) ? L"cuda" : (g_be_id == 3) ? L"vulkan" : L"cpu";

    while (!g_stop) {
        // Read one chunk (200ms)
        int chunk = in_rate / 5;
        float *chunk_buf = (float *)malloc((size_t)chunk * in_ch * sizeof(float));
        int n = wasapi_capture_read(cap, chunk_buf, chunk);
        if (n <= 0) { free(chunk_buf); Sleep(10); continue; }

        // Accumulate into batch
        int copy_n = n;
        if (batch_count + copy_n > batch_samples)
            copy_n = batch_samples - batch_count;
        // Interleave → deinterleave (take first channel)
        for (int i = 0; i < copy_n; i++)
            batch[batch_count + i] = chunk_buf[i * in_ch];
        batch_count += copy_n;
        free(chunk_buf);

        // If batch full, convert
        if (batch_count >= batch_samples) {
            // Write input WAV
            if (wav_write_16(in_path, batch, batch_count, in_rate) == 0) {
                // Build CLI command
                wchar_t cli[300];
                swprintf(cli, 300, L"%sspartacvc.exe", g_exe_dir);
                wchar_t cmd[2048];
                swprintf(cmd, 2048,
                    L"\"%s\" \"%s\" --voice \"%s\" --hubert \"%s\" --rmvpe \"%s\" --backend %s --out \"%s\"",
                    cli, in_path, g_model, g_hubert, g_rmvpe, be_str, out_path);

                STARTUPINFOW si = { sizeof(si) };
                PROCESS_INFORMATION pi = {0};
                si.dwFlags = STARTF_USESHOWWINDOW;
                si.wShowWindow = SW_HIDE;
                if (CreateProcessW(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW,
                                   g_exe_dir, NULL, &si, &pi)) {
                    WaitForSingleObject(pi.hProcess, 60000);
                    DWORD ec = 0;
                    GetExitCodeProcess(pi.hProcess, &ec);
                    CloseHandle(pi.hProcess);
                    CloseHandle(pi.hThread);

                    if (ec == 0) {
                        // Read output and play
                        wav_buf_t out_wav;
                        if (wav_read_f(out_path, &out_wav) == 0) {
                            // Resample to output rate if needed (simple linear)
                            if (out_wav.sr != out_rate) {
                                int64_t out_n = (int64_t)((double)out_wav.n * out_rate / out_wav.sr);
                                float *resampled = (float *)malloc((size_t)out_n * sizeof(float));
                                for (int64_t i = 0; i < out_n; i++) {
                                    double src_pos = (double)i * out_wav.sr / out_rate;
                                    int64_t i0 = (int64_t)src_pos;
                                    int64_t i1 = i0 + 1;
                                    double frac = src_pos - i0;
                                    if (i1 >= out_wav.n) i1 = out_wav.n - 1;
                                    resampled[i] = out_wav.data[i0] * (1.0 - frac) + out_wav.data[i1] * frac;
                                }
                                free(out_wav.data);
                                out_wav.data = resampled;
                                out_wav.n = out_n;
                                out_wav.sr = out_rate;
                            }
                            // Write to render (in chunks to avoid buffer overflow)
                            int64_t played = 0;
                            int write_chunk = out_rate / 5; // 200ms
                            while (played < out_wav.n && !g_stop) {
                                int wn = (int)((out_wav.n - played) < write_chunk ?
                                               (out_wav.n - played) : write_chunk);
                                wasapi_render_write(rnd, out_wav.data + played, wn);
                                played += wn;
                            }
                        }
                    }
                }
            }
            batch_count = 0;
        }
    }

    free(batch);
    wasapi_render_stop(rnd);
    wasapi_capture_close(cap);
    wasapi_render_close(rnd);
    wasapi_co_uninit();
    g_running = 0;
    return 0;
}

// --- Public API ---

void * spcv_rt_start(const wchar_t *model_path, int be_id, int gpu_id,
                     const wchar_t *in_dev, const wchar_t *out_dev,
                     const wchar_t *hubert_path, const wchar_t *rmvpe_path) {
    (void)gpu_id;

    if (g_running) return g_thread;

    // Store config
    if (model_path) wcscpy_s(g_model, 300, model_path);
    if (hubert_path) wcscpy_s(g_hubert, 300, hubert_path);
    if (rmvpe_path) wcscpy_s(g_rmvpe, 300, rmvpe_path);
    if (in_dev) wcscpy_s(g_in_dev, 260, in_dev);
    else g_in_dev[0] = 0;
    if (out_dev) wcscpy_s(g_out_dev, 260, out_dev);
    else g_out_dev[0] = 0;
    g_be_id = be_id;

    // Get exe directory
    GetModuleFileNameW(NULL, g_exe_dir, 260);
    wchar_t *slash = wcsrchr(g_exe_dir, L'\\');
    if (slash) *(slash + 1) = 0;

    g_stop = 0;
    g_thread = CreateThread(NULL, 0, rt_worker, NULL, 0, NULL);
    return g_thread;
}

void spcv_rt_stop(void *thread) {
    (void)thread;
    g_stop = 1;
    if (g_thread) {
        WaitForSingleObject(g_thread, 5000);
        CloseHandle(g_thread);
        g_thread = NULL;
    }
    g_running = 0;
}

int spcv_rt_is_running(void) {
    return g_running;
}

// pipeline_rt.c — Real-time RVC pipeline worker thread.
//
// Thread loop:
//   1. Read chunk from WASAPI capture (blocking)
//   2. Run RVC pipeline (HuBERT → mel → RMVPE → enc → flow+dec)
//   3. Write result to WASAPI render
//
// Chunk size: ~200ms of audio at the capture rate.
// The pipeline expects 24kHz input for HuBERT; we resample if needed.
//
// Uses the same pipeline code as the CLI (onnx.c, rvc_infer.c, ggml_pipeline.c).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wasapi.h"
#include "pipeline_rt.h"

// Forward declarations from the pipeline code
// (these are in onnx.c, rvc_infer.c, ggml_pipeline.c)
// For now, the real-time path calls spartacvc.exe via CreateProcess
// for file conversion. True real-time requires linking all pipeline code.

static volatile int g_stop = 0;
static volatile int g_running = 0;
static HANDLE g_thread = NULL;

// --- File conversion: call spartacvc.exe with the right args ---

int spcv_convert_file(const wchar_t *input_wav, const wchar_t *output_wav,
                      const wchar_t *model, int be_id,
                      const wchar_t *hubert, const wchar_t *rmvpe) {
    // Build command line: spartacvc.exe input.wav --voice model --backend X --out output.wav ...
    wchar_t cmd[2048];
    const wchar_t *be_str = (be_id == 1) ? L"cuda" : (be_id == 3) ? L"vulkan" : L"cpu";

    swprintf(cmd, 2048,
        L"spartacvc.exe \"%s\" --voice \"%s\" --backend %s --out \"%s\" --hubert \"%s\" --rmvpe \"%s\"",
        input_wav, model, be_str, output_wav, hubert, rmvpe);

    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = {0};
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE,
                        CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "[rt] CreateProcess failed: %lu\n", GetLastError());
        return -1;
    }

    WaitForSingleObject(pi.hProcess, 120000); // 2 min timeout
    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (exit_code == 0) ? 0 : -1;
}

// --- Real-time worker (placeholder: uses file-conversion in a loop) ---
// TODO: Replace with inline pipeline call for true real-time.

static DWORD WINAPI rt_worker(LPVOID arg) {
    (void)arg;
    g_running = 1;

    // Open WASAPI
    wasapi_co_init();
    wasapi_cap_t *cap = wasapi_capture_open(NULL);
    wasapi_render_t *rnd = wasapi_render_open(NULL);

    if (!cap || !rnd) {
        fprintf(stderr, "[rt] Failed to open audio devices\n");
        if (cap) wasapi_capture_close(cap);
        if (rnd) wasapi_render_close(rnd);
        wasapi_co_uninit();
        g_running = 0;
        return 1;
    }

    int in_rate = wasapi_capture_rate(cap);
    int in_ch = wasapi_capture_channels(cap);
    int out_rate = wasapi_render_rate(rnd);
    int out_ch = wasapi_render_channels(rnd);

    fprintf(stderr, "[rt] In: %d Hz x%d ch, Out: %d Hz x%d ch\n",
            in_rate, in_ch, out_rate, out_ch);

    wasapi_render_start(rnd);

    // Chunk: 200ms at input rate
    int chunk_samples = in_rate / 5; // 200ms
    float *in_buf = (float *)malloc((size_t)chunk_samples * in_ch * sizeof(float));
    float *out_buf = (float *)malloc((size_t)chunk_samples * out_ch * sizeof(float));
    // Pass-through: output same duration as input.
    // Real RVC: T frames in → T*hop samples out (same duration).

    while (!g_stop) {
        int n = wasapi_capture_read(cap, in_buf, chunk_samples);
        if (n <= 0) { Sleep(10); continue; }

        // TODO: Run RVC pipeline on in_buf → out_buf
        // For now, just pass through (identity) to test audio path
        int out_n = n; // same duration
        if (out_ch != in_ch) {
            // Simple channel copy (take first channel)
            for (int i = 0; i < n; i++)
                out_buf[i] = in_buf[i % in_ch];
        } else {
            memcpy(out_buf, in_buf, (size_t)n * in_ch * sizeof(float));
        }

        wasapi_render_write(rnd, out_buf, out_n);
    }

    wasapi_render_stop(rnd);
    free(in_buf);
    free(out_buf);
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
    (void)model_path; (void)be_id; (void)gpu_id;
    (void)in_dev; (void)out_dev; (void)hubert_path; (void)rmvpe_path;

    if (g_running) return g_thread;

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

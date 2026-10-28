// wasapi.h — WASAPI capture and render for spartacvc real-time mode.
//
// Provides:
//   - Device enumeration (input + output)
//   - Shared-mode capture (read from mic)
//   - Shared-mode render (write to speakers / virtual cable)
//   - Simple ring-buffer handoff between capture and processing threads
//
// All functions are synchronous (blocking read/write). The caller manages
// threading.
#ifndef SPCV_WASAPI_H
#define SPCV_WASAPI_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Device info ---
typedef struct {
    wchar_t id[256];       // device ID (MMDevice ID string)
    wchar_t friendly[256]; // friendly name
    int     is_default_in;
    int     is_default_out;
} wasapi_dev_t;

// --- Capture ---
typedef struct wasapi_cap wasapi_cap_t;

/** Open a capture device (shared mode, 32-bit float, device's native rate).
 *  @param dev_id  Device ID string (from wasapi_enum_devices), or NULL for default.
 *  @return        Handle, or NULL on failure. */
wasapi_cap_t * wasapi_capture_open(const wchar_t *dev_id);

/** Read up to n samples (interleaved float32) from the capture buffer.
 *  Blocks until at least 1 sample is available.
 *  @return        Number of samples actually read, or -1 on error. */
int wasapi_capture_read(wasapi_cap_t *cap, float *buf, int n);

/** Get the capture sample rate. */
int wasapi_capture_rate(const wasapi_cap_t *cap);

/** Get the capture channel count. */
int wasapi_capture_channels(const wasapi_cap_t *cap);

/** Close and release. */
void wasapi_capture_close(wasapi_cap_t *cap);

// --- Render ---
typedef struct wasapi_render wasapi_render_t;

/** Open a render device (shared mode, 32-bit float, device's native rate).
 *  @param dev_id  Device ID string, or NULL for default output.
 *  @return        Handle, or NULL on failure. */
wasapi_render_t * wasapi_render_open(const wchar_t *dev_id);

/** Write n samples (interleaved float32) to the render buffer.
 *  Blocks until the buffer has space.
 *  @return        0 on success, -1 on error. */
int wasapi_render_write(wasapi_render_t *rnd, const float *buf, int n);

/** Get the render sample rate. */
int wasapi_render_rate(const wasapi_render_t *rnd);

/** Get the render channel count. */
int wasapi_render_channels(const wasapi_render_t *rnd);

/** Start / stop the render stream. */
int wasapi_render_start(wasapi_render_t *rnd);
int wasapi_render_stop(wasapi_render_t *rnd);

/** Close and release. */
void wasapi_render_close(wasapi_render_t *rnd);

// --- Device enumeration ---
/** Enumerate audio input devices.
 *  @param devs      Array to fill (caller allocates at least max_n).
 *  @param max_n     Size of devs array.
 *  @return          Number of devices written, or -1 on COM failure. */
int wasapi_enum_inputs(wasapi_dev_t *devs, int max_n);

/** Enumerate audio output devices. */
int wasapi_enum_outputs(wasapi_dev_t *devs, int max_n);

/** Initialize COM (call once at startup). */
int wasapi_co_init(void);

/** Uninitialize COM. */
void wasapi_co_uninit(void);

#ifdef __cplusplus
}
#endif

#endif // SPCV_WASAPI_H

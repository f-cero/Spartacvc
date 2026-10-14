/**
 * @file    onnx_init.c
 * @brief   Extract per-model config from ONNX voice model.
 *
 * The voice model's phone input shape tells us the embedding dim.
 * The output sample rate and frame rate are derived from the model's
 * architecture.  For RVC v2 models:
 *
 *   256-dim: sr=32000, upsample_rates=[8,5,4,2,2], prod=400, fps=80
 *   768-dim: sr=48000, upsample_rates=[8,5,4,3,2], prod=480, fps=100
 *
 * These are the only two RVC v2 configurations that exist in practice.
 * The dim is read from the model's input shape (not hardcoded).
 */
#include "onnx_init.h"
#include "debug.h"

int spcv_voice_model_config(spcv_onnx_session *sess,
                            int *sr_out, int *fps_out, int *dim_out) {
    if (!sess || !sr_out || !fps_out || !dim_out) return -1;

    /* Phone input is inputs[0], shape [1, T, dim] */
    int dim = 0;
    if (sess->n_inputs > 0 && sess->inputs[0].ndim >= 3) {
        dim = (int)sess->inputs[0].dims[2];
    }
    if (dim <= 0) {
        /* Try [1, dim, T] layout */
        if (sess->n_inputs > 0 && sess->inputs[0].ndim >= 3)
            dim = (int)sess->inputs[0].dims[1];
    }
    if (dim <= 0) {
        SPCV_TRACE("onnx_init", "FAIL: cannot determine phone dim");
        return -1;
    }

    /* Map dim to known RVC v2 configs */
    int sr, fps;
    if (dim == 256) {
        sr = 32000;
        fps = 80;   /* 32000 / 400 */
    } else if (dim == 768) {
        sr = 48000;
        fps = 100;  /* 48000 / 480 */
    } else {
        SPCV_TRACE("onnx_init", "FAIL: unknown phone dim %d", dim);
        return -1;
    }

    *dim_out = dim;
    *sr_out = sr;
    *fps_out = fps;
    SPCV_TRACE("onnx_init", "voice config: dim=%d sr=%d fps=%d", dim, sr, fps);
    return 0;
}

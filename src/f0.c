/**
 * @file    f0.c
 * @brief   Pitch extraction — RMVPE inference + cents decode + shift + quantize.
 *
 * RMVPE model:
 *   Input:  [1, 128, T]  (mel transposed, T must be multiple of 32)
 *   Output: [1, T, 360]  (salience over 360 cent bins)
 *
 * Cents decode (9-bin averaging):
 *   For each frame, find argmax bin, average 9 surrounding bins
 *   weighted by their cent values, convert cents -> Hz.
 *
 * Pitch shift:
 *   f0 *= 2^(semitones/12)
 *
 * Quantization (RVC standard):
 *   log-f0 in [50, 1100] Hz mapped to [1, 255]
 */
#include "f0.h"
#include "debug.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N_CENTS   360
#define N_AVG     9       /* 9-bin averaging window */
#define F0_MIN    50.0
#define F0_MAX    1100.0

/* ------------------------------------------------------------------ */
/*  Cents decode                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief  Decode RMVPE salience output to f0 in Hz.
 *
 * @param  sal       salience [n_frames, 360]
 * @param  n_frames  number of frames
 * @param  f0_out    [out] f0 in Hz (caller allocates)
 */
static void cents_decode(const float *sal, int n_frames, float *f0_out) {
    /* Cent values: 20 * i + 1997.3794084376191 for i in 0..359 */
    static float cents_table[N_CENTS];
    static int table_init = 0;
    if (!table_init) {
        for (int i = 0; i < N_CENTS; i++)
            cents_table[i] = 20.0f * i + 1997.3794084376191f;
        table_init = 1;
    }

    int half = N_AVG / 2; /* 4 */

    for (int t = 0; t < n_frames; t++) {
        const float *row = sal + t * N_CENTS;

        /* Find peak bin */
        int peak = 0;
        float max_val = row[0];
        for (int c = 1; c < N_CENTS; c++) {
            if (row[c] > max_val) { max_val = row[c]; peak = c; }
        }

        /* Confidence check */
        if (max_val <= 0.05f) {
            f0_out[t] = 0.0f;
            continue;
        }

        /* 9-bin weighted average around peak */
        int lo = peak - half;
        int hi = peak + half + 1;
        if (lo < 0) lo = 0;
        if (hi > N_CENTS) hi = N_CENTS;

        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        for (int c = lo; c < hi; c++) {
            weighted_sum += row[c] * cents_table[c];
            weight_sum += row[c];
        }

        if (weight_sum <= 0.0f) {
            f0_out[t] = 0.0f;
            continue;
        }

        float cents = weighted_sum / weight_sum;
        f0_out[t] = 10.0f * powf(2.0f, cents / 1200.0f);
    }
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int spcv_f0_extract(const float *mel, int n_frames,
                    spcv_onnx_session *rmvpe, float *f0_out) {
    if (!mel || n_frames <= 0 || !rmvpe || !f0_out) return -1;

    /* Pad time to multiple of 32 */
    int padded = 32 * ((n_frames - 1) / 32 + 1);
    int pad_amt = padded - n_frames;

    float *mel_padded = malloc(padded * 128 * sizeof(float));
    if (!mel_padded) return -1;

    /* Transpose: [time, 128] -> [128, time] (channels-first for RMVPE) */
    memset(mel_padded, 0, padded * 128 * sizeof(float));
    for (int t = 0; t < n_frames; t++)
        for (int m = 0; m < 128; m++)
            mel_padded[m * padded + t] = mel[t * 128 + m];

    SPCV_TRACE("f0", "RMVPE input: [1, 128, %d] (padded from %d)", padded, n_frames);

    /* Run RMVPE */
    spcv_onnx_input in = {
        .name = rmvpe->inputs[0].name,
        .data = mel_padded,
        .ndim = 3,
        .dims = {1, 128, padded},
        .elem_type = 1 /* FLOAT */
    };
    spcv_onnx_output out = { .name = rmvpe->outputs[0].name, .buf = NULL, .buf_size = 0, .ort_value = NULL };

    int rc = spcv_onnx_run(rmvpe, &in, 1, &out, 1);
    if (rc != SPCV_ONNX_OK) {
        SPCV_TRACE("f0", "FAIL: RMVPE run: %s", spcv_onnx_status_str(rc));
        free(mel_padded);
        return -1;
    }

    int ndim; int64_t dims[8];
    spcv_onnx_output_shape(&out, &ndim, dims);
    const float *sal = (const float *)spcv_onnx_output_data(&out);
    if (!sal) {
        SPCV_TRACE("f0", "FAIL: no output data");
        spcv_onnx_free_output(&out);
        free(mel_padded);
        return -1;
    }

    /* sal shape: [1, padded, 360] */
    int sal_frames = (ndim >= 3) ? (int)dims[1] : padded;
    SPCV_TRACE("f0", "RMVPE output: [%d] %lld %lld %lld",
               ndim, (long long)dims[0], (long long)dims[1], (long long)dims[2]);

    /* Decode cents -> Hz (only original frames) */
    cents_decode(sal, n_frames, f0_out);

    /* Stats */
    int voiced = 0;
    float fmin = 9999, fmax = 0;
    for (int t = 0; t < n_frames; t++) {
        if (f0_out[t] > 0) {
            voiced++;
            if (f0_out[t] < fmin) fmin = f0_out[t];
            if (f0_out[t] > fmax) fmax = f0_out[t];
        }
    }
    SPCV_TRACE("f0", "voiced=%d/%d range=%.1f-%.1f Hz", voiced, n_frames, fmin, fmax);

    spcv_onnx_free_output(&out);
    free(mel_padded);
    return 0;
}

void spcv_f0_shift(float *f0, int n_frames, int semitones) {
    if (semitones == 0) return;
    float factor = powf(2.0f, semitones / 12.0f);
    SPCV_TRACE("f0", "pitch shift: %+d st (x%.4f)", semitones, factor);
    for (int t = 0; t < n_frames; t++)
        if (f0[t] > 0) f0[t] *= factor;
}

void spcv_f0_quantize(const float *f0, int n_frames, int64_t *pitch_out) {
    float fmm = 1127.0f * logf(1.0f + F0_MIN / 700.0f);
    float fmx = 1127.0f * logf(1.0f + F0_MAX / 700.0f);

    for (int t = 0; t < n_frames; t++) {
        if (f0[t] <= 0) {
            pitch_out[t] = 1;
            continue;
        }
        float fm = 1127.0f * logf(1.0f + f0[t] / 700.0f);
        float q = (fm - fmm) * 254.0f / (fmx - fmm) + 1.0f;
        if (q < 1) q = 1;
        if (q > 255) q = 255;
        pitch_out[t] = (int64_t)lroundf(q);
    }
}

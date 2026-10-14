/**
 * @file    mel.c
 * @brief   Mel spectrogram — implementation.
 *
 * Hand-rolled FFT + mel filterbank. No external dependencies.
 *
 * Algorithm:
 *   1. Pad audio with n_fft/2 on each side (reflect mode)
 *   2. Frame: hop=160, win=1024, Hanning window
 *   3. FFT: real FFT via complex FFT (n_fft/2+1 bins)
 *   4. Power spectrum: |X[k]|^2
 *   5. Mel filterbank: 128 triangular filters
 *   6. Log: log(clip(mel, 1e-5, inf))
 */
#include "mel.h"
#include "debug.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N_FFT   1024
#define HOP     160
#define SR      16000

/* ------------------------------------------------------------------ */
/*  Hanning window (computed once)                                     */
/* ------------------------------------------------------------------ */

static float g_window[N_FFT];
static int   g_window_init = 0;

static void init_window(void) {
    if (g_window_init) return;
    for (int i = 0; i < N_FFT; i++)
        g_window[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / N_FFT));
    g_window_init = 1;
}

/* ------------------------------------------------------------------ */
/*  Mel filterbank                                                     */
/* ------------------------------------------------------------------ */

static float g_mel_fb[128][N_FFT / 2 + 1];
static int   g_mel_init = 0;

static double hz2mel(double hz) { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel2hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

static void init_mel_fb(int n_mels) {
    if (g_mel_init) return;
    double fmin = 30.0, fmax = 8000.0;
    double low = hz2mel(fmin), high = hz2mel(fmax);
    int nfbin = N_FFT / 2 + 1;

    /* Mel points (n_mels + 2 for triangular edges) */
    double *mp = malloc((n_mels + 2) * sizeof(double));
    double *hz = malloc((n_mels + 2) * sizeof(double));
    int *bins = malloc((n_mels + 2) * sizeof(int));

    for (int i = 0; i < n_mels + 2; i++) {
        mp[i] = low + (high - low) * i / (n_mels + 1);
        hz[i] = mel2hz(mp[i]);
        bins[i] = (int)floor((N_FFT / 2.0 + 1.0) * hz[i] / (SR / 2.0));
    }

    for (int i = 0; i < n_mels; i++) {
        memset(g_mel_fb[i], 0, sizeof(g_mel_fb[i]));
        int l = bins[i], c = bins[i + 1], r = bins[i + 2];
        for (int j = l; j < c; j++)
            if (c > l) g_mel_fb[i][j] = (float)((j - l) / (double)(c - l));
        for (int j = c; j < r; j++)
            if (r > c) g_mel_fb[i][j] = (float)((r - j) / (double)(r - c));
    }

    free(mp); free(hz); free(bins);
    g_mel_init = 1;
}

/* ------------------------------------------------------------------ */
/*  Real FFT (iterative Cooley-Tukey, in-place)                        */
/* ------------------------------------------------------------------ */

typedef struct { float re, im; } spcv_cpx;

/**
 * @brief  In-place iterative FFT (radix-2, requires n to be power of 2).
 * @param  a      complex array of length n
 * @param  n      length (power of 2)
 * @param  invert 0 = forward, 1 = inverse
 */
static void fft(spcv_cpx *a, int n, int invert) {
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { spcv_cpx tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
    }
    /* Butterfly */
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * 3.14159265358979 * (invert ? -1.0 : 1.0) / len;
        spcv_cpx wlen = { (float)cos(ang), (float)sin(ang) };
        for (int i = 0; i < n; i += len) {
            spcv_cpx w = { 1.0f, 0.0f };
            for (int j = 0; j < len / 2; j++) {
                spcv_cpx u = a[i + j];
                spcv_cpx v = { a[i+j+len/2].re * w.re - a[i+j+len/2].im * w.im,
                               a[i+j+len/2].re * w.im + a[i+j+len/2].im * w.re };
                a[i+j]       = (spcv_cpx){ u.re + v.re, u.im + v.im };
                a[i+j+len/2] = (spcv_cpx){ u.re - v.re, u.im - v.im };
                w = (spcv_cpx){ w.re * wlen.re - w.im * wlen.im,
                                w.re * wlen.im + w.im * wlen.re };
            }
        }
    }
    if (invert) {
        for (int i = 0; i < n; i++) { a[i].re /= n; a[i].im /= n; }
    }
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

int64_t spcv_mel_n_frames(int64_t n_samples) {
    if (n_samples <= 0) return 0;
    int64_t pad = N_FFT / 2;
    int64_t padded = n_samples + 2 * pad;
    return 1 + (padded - N_FFT) / HOP;
}

int spcv_mel_compute(const float *audio, int64_t n_samples,
                     float *mel_out, int n_mels) {
    if (!audio || n_samples <= 0 || !mel_out || n_mels <= 0 || n_mels > 128)
        return -1;

    init_window();
    init_mel_fb(n_mels);

    int64_t n_frames = spcv_mel_n_frames(n_samples);
    int nfbin = N_FFT / 2 + 1;

    SPCV_TRACE("mel", "computing: %lld samples -> %lld frames x %d mels",
               (long long)n_samples, (long long)n_frames, n_mels);

    /* Allocate scratch */
    spcv_cpx *fft_buf = malloc(N_FFT * sizeof(spcv_cpx));
    float *power = malloc(nfbin * sizeof(float));
    if (!fft_buf || !power) { free(fft_buf); free(power); return -1; }

    int64_t pad = N_FFT / 2;

    for (int64_t f = 0; f < n_frames; f++) {
        int64_t start = f * HOP;

        /* Load frame with Hanning window */
        for (int i = 0; i < N_FFT; i++) {
            int64_t idx = start + i - pad;
            float val;
            if (idx < 0) {
                /* Reflect padding */
                int64_t r = -idx;
                val = (r < n_samples) ? audio[r] : 0.0f;
            } else if (idx >= n_samples) {
                int64_t r = 2 * n_samples - 1 - idx;
                val = (r >= 0 && r < n_samples) ? audio[r] : 0.0f;
            } else {
                val = audio[idx];
            }
            fft_buf[i].re = val * g_window[i];
            fft_buf[i].im = 0.0f;
        }

        /* FFT */
        fft(fft_buf, N_FFT, 0);

        /* Power spectrum */
        for (int k = 0; k < nfbin; k++)
            power[k] = fft_buf[k].re * fft_buf[k].re + fft_buf[k].im * fft_buf[k].im;

        /* Mel filterbank + log */
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < nfbin; k++)
                sum += g_mel_fb[m][k] * power[k];
            mel_out[f * n_mels + m] = logf(sum > 1e-5f ? sum : 1e-5f);
        }
    }

    free(fft_buf);
    free(power);
    SPCV_TRACE("mel", "done: %lld frames", (long long)n_frames);
    return (int)n_frames;
}

/**
 * @file    ggml_pipeline.c
 * @brief   ggml RVC pipeline: flow + decoder on GPU (encoder TBD).
 *
 * Uses gguf_load.c for GGUF→ggml backend.
 * Flow: 4× RCL (WaveNet 3-layer) with conv_1d on GPU.
 * Decoder: conv_pre + 4×(upsample+resblocks) + conv_post.
 */
#include "ggml_pipeline.h"
#include "gguf_load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include <ggml-alloc.h>

/* 10-second timeout guard: call TMO_CHECK(label) after each major op */
static double tmo_start = 0;
#define TMO_BEGIN() do { tmo_start = (double)clock()/CLOCKS_PER_SEC; } while(0)
#define TMO_CHECK(label) do { double dt = (double)clock()/CLOCKS_PER_SEC - tmo_start; \
    if (dt > 10.0) { fprintf(stderr, "[TIMEOUT] %s took %.1fs (>10s) — ABORT\n", label, dt); exit(99); } \
    else if (dt > 1.0) fprintf(stderr, "[tmo] %s: %.2fs\n", label, dt); } while(0)

#define FC_DIM  192
#define UP_DIM  512
#define GIN     256
#define NU      4
#define NUP     480

static const int up_K[4]      = {24, 20, 4, 4};
static const int up_stride[4] = {12, 10, 2, 2};
static const int up_OC[4]     = {256, 128, 64, 32};
static const int up_pad[4]    = {6, 5, 1, 1};
static const int rb_K[3]      = {3, 7, 11};
static const int rb_dil[3]    = {1, 3, 5};

/* ---- helpers (same as test files) ---- */

/* Input is already [C, L] (C-fast, L fastest) = same as ggml [L, C] (ne[0]=L fastest).
 * Just copy. Do NOT transpose. */
static float *host_t_ct(const float *in, int C, int L) {
    float *out = (float *)malloc((size_t)C * L * 4);
    memcpy(out, in, (size_t)C * L * 4);
    return out;
}

/* Manual CPU conv1d with dilation, using fused GGUF weights [K, IC, OC].
 * in: [IC, L] (IC-slow, L-fast).  out: [OC, L]. */
static int cpu_conv1d(spcv_ggml_t *g, const char *wname, const char *bname,
                      const float *in, int IC, int L, int K, int dil,
                      float *out, int OC) {
    int64_t nn;
    size_t wbytes = (size_t)K * IC * OC * 4;
    float *W = (float *)malloc(wbytes);
    if (!W) return -1;
    if (spcv_ggml_get_tensor(g, wname, W, wbytes, &nn) != 0) { free(W); return -1; }
    float *B = NULL;
    if (bname) {
        B = (float *)malloc(OC * 4);
        if (spcv_ggml_get_tensor(g, bname, B, OC * 4, &nn) != 0) { free(W); free(B); return -1; }
    }
    int pad = (K - 1) * dil / 2;
    for (int co = 0; co < OC; co++) {
        for (int l = 0; l < L; l++) {
            float s = B ? B[co] : 0.0f;
            for (int ci = 0; ci < IC; ci++) {
                for (int k = 0; k < K; k++) {
                    int p = l - pad + k * dil;
                    if (p >= 0 && p < L)
                        s += W[k + ci * K + co * IC * K] * in[ci * L + p];
                }
            }
            out[co * L + l] = s;
        }
    }
    free(W);
    free(B);
    return 0;
}

static int gpu_conv1d(spcv_ggml_t *g, const char *wname, const char *bname,
                      const float *in, int IC, int L, int K, int dil,
                      float *out, int OC) {
    struct ggml_tensor *W = spcv_ggml_tensor(g, wname);
    if (!W) { fprintf(stderr, "[ggml] weight '%s' not found\n", wname); return -1; }
    float *xt_host = host_t_ct(in, IC, L);
    struct ggml_context *ctx = spcv_ggml_ctx(g);
    struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);
    ggml_set_name(xin, "xin");
    int pad = (K - 1) * dil / 2;
    struct ggml_tensor *co = ggml_conv_1d(ctx, W, xin, 1, pad, dil);
    if (bname) {
        struct ggml_tensor *bias = spcv_ggml_tensor(g, bname);
        if (bias) {
            struct ggml_tensor *br = ggml_reshape_2d(ctx, bias, 1, OC);
            co = ggml_add(ctx, co, br);
        }
    }
    ggml_set_name(co, "cout");
    spcv_ggml_set_input(g, "xin", xt_host, (size_t)IC * L * 4);
    size_t actual_bytes = ggml_nbytes(co);
    size_t expected_bytes = (size_t)OC * (L + 2 * pad - dil * (K - 1)) * 4;
    if (actual_bytes > expected_bytes) {
        fprintf(stderr, "[conv1d] FATAL: actual(%zu) > expected(%zu)\n", actual_bytes, expected_bytes);
        free(xt_host); return -1;
    }
    if (spcv_ggml_compute(g, co, out, actual_bytes) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ggml] conv1d '%s' fail\n", wname);
        free(xt_host); return -1;
    }
    free(xt_host);
    if (actual_bytes < expected_bytes)
        memset((unsigned char *)out + actual_bytes, 0, expected_bytes - actual_bytes);
    return 0;
}

static int gpu_conv_t1d(spcv_ggml_t *g, const char *wname, const char *bname,
                        const float *in, int IC, int L,
                        int K, int stride, int pad, float *out, int OC) {
    struct ggml_tensor *W = spcv_ggml_tensor(g, wname);
    if (!W) { fprintf(stderr, "[ggml] weight '%s' not found\n", wname); return -1; }
    float *xt_host = host_t_ct(in, IC, L);
    struct ggml_context *ctx = spcv_ggml_ctx(g);
    struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);
    ggml_set_name(xin, "xin");
    struct ggml_tensor *co = ggml_conv_transpose_1d(ctx, W, xin, stride, 0, 1);
    if (bname) {
        struct ggml_tensor *bias = spcv_ggml_tensor(g, bname);
        if (bias) {
            struct ggml_tensor *br = ggml_reshape_2d(ctx, bias, 1, OC);
            co = ggml_add(ctx, co, br);
        }
    }
    ggml_set_name(co, "cout");
    spcv_ggml_set_input(g, "xin", xt_host, (size_t)IC * L * 4);
    int full_L = (L - 1) * stride + K;
    float *full_out = (float *)malloc((size_t)OC * full_L * 4);
    if (spcv_ggml_compute(g, co, full_out, (size_t)OC * full_L * 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ggml] conv_t1d '%s' fail\n", wname);
        free(xt_host); free(full_out); return -1;
    }
    int out_L = full_L - 2 * pad;
    for (int c = 0; c < OC; c++)
        for (int t = 0; t < out_L; t++)
            out[t + c * out_L] = full_out[(t + pad) + c * full_L];
    free(xt_host); free(full_out); return 0;
}

static int gpu_leaky_relu(spcv_ggml_t *g, float *data, int n) {
    struct ggml_context *ctx = spcv_ggml_ctx(g);
    struct ggml_tensor *xin = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(xin, "xin");
    struct ggml_tensor *out = ggml_leaky_relu(ctx, xin, 0.1f, false);
    ggml_set_name(out, "cout");
    spcv_ggml_set_input(g, "xin", data, (size_t)n * 4);
    if (spcv_ggml_compute(g, out, data, (size_t)n * 4) != GGML_STATUS_SUCCESS) return -1;
    return 0;
}

/* ---- flow ---- */

static int ggml_flow(spcv_ggml_t *g, const float *m_p, const float *logs_p,
                     const float *g_emb, int T, float *z) {
    /* VAE reparameterization: z = m_p + exp(logs_p) * noise * 0.66666 */
    if (getenv("SPCV_GGML_USE_REF_Z")) {
        /* Load reference z from pure-C pipeline for direct comparison */
        FILE *f = fopen("scratch/vae_z.bin", "rb");
        if (!f) { fprintf(stderr, "[ggml] cannot open scratch/vae_z.bin\n"); return -1; }
        if (fread(z, sizeof(float), (size_t)FC_DIM * T, f) != (size_t)FC_DIM * T) {
            fprintf(stderr, "[ggml] vae_z.bin too small\n"); fclose(f); return -1;
        }
        fclose(f);
        fprintf(stderr, "[ggml] using reference vae_z from pure-C\n");
    } else {
        srand(42);
        for (int c = 0; c < FC_DIM; c++)
            for (int t = 0; t < T; t++) {
                float u1 = (float)((rand() + 1) / ((double)RAND_MAX + 1.0));
                float u2 = (float)((rand() + 1) / ((double)RAND_MAX + 1.0));
                float noise = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
                z[c * T + t] = m_p[c * T + t] + expf(logs_p[c * T + t]) * noise * 0.66666f;
            }
    }
    if (getenv("SPCV_GGML_DUMP")) {
        FILE *f = fopen("scratch/ggml_vae_z.bin", "wb");
        if (f) { fwrite(z, sizeof(float), (size_t)FC_DIM * T, f); fclose(f); }
    }
    /* flip */
    float *tmp = (float *)malloc((size_t)FC_DIM * T * 4);
    for (int c = 0; c < FC_DIM; c++)
        memcpy(&tmp[c * T], &z[(FC_DIM-1-c) * T], (size_t)T * 4);
    memcpy(z, tmp, (size_t)FC_DIM * T * 4);
    free(tmp);

    int flow_idx[4] = {6, 4, 2, 0};
    for (int fi = 0; fi < 4; fi++) {
        int idx = flow_idx[fi];
        char nm[256], bn[256];
        float *h = (float *)malloc((size_t)FC_DIM * T * 4);
        fprintf(stderr, "[perf] flow layer %d (idx=%d) pre\n", fi, idx);

        snprintf(nm, sizeof(nm), "flow.flows.%d.pre.weight", idx);
        snprintf(bn, sizeof(bn), "flow.flows.%d.pre.bias", idx);
        if (gpu_conv1d(g, nm, bn, z, 96, T, 1, 1, h, FC_DIM) != 0) { free(h); return -1; }
        fprintf(stderr, "[perf] flow layer %d pre done\n", fi);
        if (getenv("SPCV_GGML_DUMP") && fi == 0) {
            FILE *f = fopen("scratch/ggml_fb0_pre.bin", "wb");
            if (f) { fwrite(h, sizeof(float), (size_t)FC_DIM * T, f); fclose(f); }
        }

        int CC = 2 * FC_DIM * 3;
        float *cond = (float *)malloc(CC * 4);
        snprintf(nm, sizeof(nm), "flow.flows.%d.enc.cond_layer.weight", idx);
        snprintf(bn, sizeof(bn), "flow.flows.%d.enc.cond_layer.bias", idx);
        if (gpu_conv1d(g, nm, bn, g_emb, GIN, 1, 1, 1, cond, CC) != 0) { free(h); free(cond); return -1; }

        /* WaveNet: h=running residual, output=skip accumulator */
        float *wn_out = (float *)calloc((size_t)FC_DIM * T, 4);
        for (int j = 0; j < 3; j++) {
            /* in_layer: conv(h) -> x_in [2*FC, T] */
            float *x_in = (float *)malloc((size_t)2 * FC_DIM * T * 4);
            snprintf(nm, sizeof(nm), "flow.flows.%d.enc.in_layers.%d.weight", idx, j);
            snprintf(bn, sizeof(bn), "flow.flows.%d.enc.in_layers.%d.bias", idx, j);
            if (gpu_conv1d(g, nm, bn, h, FC_DIM, T, 5, 1, x_in, 2 * FC_DIM) != 0) { free(h); free(cond); free(x_in); free(wn_out); return -1; }
            /* fused_add_tanh_sigmoid: acts = tanh(x_in[:FC]+cv1) / (1+exp(-(x_in[FC:]+cv2))) */
            float *acts = (float *)malloc((size_t)FC_DIM * T * 4);
            for (int c = 0; c < FC_DIM; c++) {
                float cv1 = cond[j * 2 * FC_DIM + c];
                float cv2 = cond[j * 2 * FC_DIM + FC_DIM + c];
                for (int t = 0; t < T; t++) {
                    float a = x_in[c * T + t] + cv1;
                    float b = x_in[(FC_DIM+c) * T + t] + cv2;
                    acts[c * T + t] = tanhf(a) / (1.0f + expf(-b));
                }
            }
            free(x_in);
            /* res_skip: conv(acts) -> res_skip */
            int rs_out = (j < 2) ? 2 * FC_DIM : FC_DIM;
            float *res_skip = (float *)malloc((size_t)rs_out * T * 4);
            snprintf(nm, sizeof(nm), "flow.flows.%d.enc.res_skip_layers.%d.weight", idx, j);
            snprintf(bn, sizeof(bn), "flow.flows.%d.enc.res_skip_layers.%d.bias", idx, j);
            if (gpu_conv1d(g, nm, bn, acts, FC_DIM, T, 1, 1, res_skip, rs_out) != 0) { free(h); free(cond); free(acts); free(res_skip); free(wn_out); return -1; }
            free(acts);
            /* accumulate */
            if (j < 2) {
                for (int c = 0; c < FC_DIM; c++)
                    for (int t = 0; t < T; t++) {
                        h[c * T + t] += res_skip[c * T + t];
                        wn_out[c * T + t] += res_skip[(FC_DIM+c) * T + t];
                    }
            } else {
                for (int c = 0; c < FC_DIM; c++)
                    for (int t = 0; t < T; t++)
                        wn_out[c * T + t] += res_skip[c * T + t];
            }
            free(res_skip);
        }
        /* WaveNet output = wn_out (NOT h) */
        memcpy(h, wn_out, (size_t)FC_DIM * T * 4);
        free(wn_out);
        if (getenv("SPCV_GGML_DUMP") && fi == 0) {
            FILE *f = fopen("scratch/ggml_fb0_wn.bin", "wb");
            if (f) { fwrite(h, sizeof(float), (size_t)FC_DIM * T, f); fclose(f); }
        }

        float *m_out = (float *)malloc((size_t)96 * T * 4);
        snprintf(nm, sizeof(nm), "flow.flows.%d.post.weight", idx);
        snprintf(bn, sizeof(bn), "flow.flows.%d.post.bias", idx);
        if (gpu_conv1d(g, nm, bn, h, FC_DIM, T, 1, 1, m_out, 96) != 0) { free(h); free(cond); free(m_out); return -1; }
        free(h);
        for (int c = 0; c < 96; c++)
            for (int t = 0; t < T; t++)
                z[(96+c) * T + t] -= m_out[c * T + t];
        free(m_out);

        if (fi < 3) {
            tmp = (float *)malloc((size_t)FC_DIM * T * 4);
            for (int c = 0; c < FC_DIM; c++)
                memcpy(&tmp[c * T], &z[(FC_DIM-1-c) * T], (size_t)T * 4);
            memcpy(z, tmp, (size_t)FC_DIM * T * 4);
            free(tmp);
        }
        free(cond);
    }
    return 0;
}

/* ---- decoder ---- */

static int ggml_decoder(spcv_ggml_t *g, const float *z, int T,
                        const float *g_emb, const float *f0, float *out) {
    int OL = T * NUP;
    int sr = 48000;

    TMO_BEGIN();
    /* --- har_source: SineGen + m_source (CPU) --- */
    float *har_source = (float *)malloc(OL * 4);
    {
        float mw = 1.0f, mb = 0.0f;
        int64_t nn;
        float wbuf[1], bbuf[1];
        if (spcv_ggml_get_tensor(g, "dec.m_source.l_linear.weight", wbuf, 4, &nn) == 0) mw = wbuf[0];
        if (spcv_ggml_get_tensor(g, "dec.m_source.l_linear.bias", bbuf, 4, &nn) == 0) mb = bbuf[0];
        int no_noise = (getenv("SPCV_NO_NOISE") != NULL);
        uint32_t rng = 54321;
        double phase = 0.0;
        for (int t = 0; t < OL; t++) {
            int frame = t / NUP;
            if (frame >= T) frame = T - 1;
            float fv = f0 ? f0[frame] : 0.0f;
            float uv = (fv > 0.0f) ? 1.0f : 0.0f;
            if (uv > 0.0f) { phase += (double)fv / sr; phase -= floor(phase); }
            double clean = uv * (0.1 * sin(2.0 * 3.14159265358979 * phase));
            double noise = 0.0;
            if (!no_noise) {
                double amp = uv * 0.003 + (1.0 - uv) * (0.1 / 3.0);
                rng = rng * 1103515245u + 12345u;
                float u1 = (float)((rng >> 8) & 0xFFFFF) / (float)(1 << 20);
                rng = rng * 1103515245u + 12345u;
                float u2 = (float)((rng >> 8) & 0xFFFFF) / (float)(1 << 20);
                if (u1 < 1e-10f) u1 = 1e-10f;
                noise = amp * (double)sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
            }
            har_source[t] = (float)tanh((double)mw * (clean + noise) + (double)mb);
        }
    }

    TMO_CHECK("har_source");
    int cur_C = UP_DIM, cur_L = T;
    fprintf(stderr, "[dec] har_source done, OL=%d\n", OL);
    TMO_BEGIN();
    float *d = (float *)malloc((size_t)UP_DIM * T * 4);
    if (gpu_conv1d(g, "dec.conv_pre.weight", "dec.conv_pre.bias", z, FC_DIM, T, 7, 1, d, UP_DIM) != 0) {
        free(d); free(har_source); return -1;
    }
    TMO_CHECK("conv_pre");
    { float rms=0; for(int i=0;i<UP_DIM*T;i++) rms+=d[i]*d[i]; fprintf(stderr, "[dec] conv_pre RMS=%.6f\n", sqrtf(rms/(UP_DIM*T))); }
    /* speaker embedding conditioning */
    if (g_emb) {
        float *wg = (float *)malloc((size_t)UP_DIM * GIN * 4);
        float *bg = (float *)malloc(UP_DIM * 4);
        int64_t nn;
        if (spcv_ggml_get_tensor(g, "dec.cond.weight", wg, (size_t)UP_DIM * GIN * 4, &nn) != 0) {
            free(d); free(wg); free(bg); free(har_source); return -1;
        }
        spcv_ggml_get_tensor(g, "dec.cond.bias", bg, (size_t)UP_DIM * 4, &nn);
        for (int c = 0; c < UP_DIM; c++) {
            float v = bg[c];
            for (int i = 0; i < GIN; i++) v += wg[c * GIN + i] * g_emb[i];
            for (int t = 0; t < T; t++) d[c * T + t] += v;
        }
        free(wg); free(bg);
    }

    for (int u = 0; u < NU; u++) {
        fprintf(stderr, "[dec] stage %d C=%d L=%d\n", u, cur_C, cur_L);
        gpu_leaky_relu(g, d, cur_C * cur_L);
        /* upsample: ConvTranspose1d */
        int nC = up_OC[u], nL = cur_L * up_stride[u];
        float *d2 = (float *)malloc((size_t)nC * nL * 4);
        char wn[256], bn[256];
        snprintf(wn, sizeof(wn), "dec.ups.%d.weight", u);
        snprintf(bn, sizeof(bn), "dec.ups.%d.bias", u);
        TMO_BEGIN();
        if (gpu_conv_t1d(g, wn, bn, d, cur_C, cur_L, up_K[u], up_stride[u], up_pad[u], d2, nC) != 0) {
            free(d); free(d2); free(har_source); return -1;
        }
        TMO_CHECK("upsample");
        free(d); d = d2;
        cur_C = nC; cur_L = nL;
        { float rms=0; for(int i=0;i<cur_C*cur_L;i++) rms+=d[i]*d[i]; fprintf(stderr, "[dec] stage %d upsample RMS=%.6f C=%d L=%d\n", u, sqrtf(rms/(cur_C*cur_L)), cur_C, cur_L); }

        /* noise_conv: Conv1d(har_source, K, stride) -> [nC, nL], add to d (manual CPU) */
        {
            int nK, stride;
            if (u + 1 < NU) {
                stride = 1;
                for (int i = u + 1; i < NU; i++) stride *= up_stride[i];
                nK = stride * 2;
            } else { nK = 1; stride = 1; }
            char nm[128], bnm[128];
            snprintf(nm, sizeof(nm), "dec.noise_convs.%d.weight", u);
            snprintf(bnm, sizeof(bnm), "dec.noise_convs.%d.bias", u);
            int64_t nn;
            float *nw = (float *)malloc((size_t)nK * nC * 4);
            float *nb = (float *)malloc(nC * 4);
            if (spcv_ggml_get_tensor(g, nm, nw, (size_t)nK * nC * 4, &nn) == 0 &&
                spcv_ggml_get_tensor(g, bnm, nb, (size_t)nC * 4, &nn) == 0) {
                int ns_L = (OL - nK) / stride + 1;
                int use_L = (ns_L < nL) ? ns_L : nL;
                for (int c = 0; c < nC; c++) {
                    for (int t = 0; t < use_L; t++) {
                        float v = nb[c];
                        int pos = t * stride;
                        for (int k = 0; k < nK; k++)
                            v += nw[k + c * nK] * har_source[pos + k];
                        d[c * nL + t] += v;
                    }
                }
            }
            free(nw); free(nb);
            { float rms=0; for(int i=0;i<cur_C*cur_L;i++) rms+=d[i]*d[i]; fprintf(stderr, "[dec] stage %d noise_conv RMS=%.6f\n", u, sqrtf(rms/(cur_C*cur_L))); }
        }

        /* 3 resblocks, averaged. Each resblock: 3 dilation pairs with residual. */
        float *sum = (float *)calloc((size_t)cur_C * cur_L, 4);
        for (int rb = 0; rb < 3; rb++) {
            int rb_idx = u * 3 + rb;
            int K = rb_K[rb];
            fprintf(stderr, "[dec] stage %d resblock %d K=%d\n", u, rb_idx, K);
            float *x = (float *)malloc((size_t)cur_C * cur_L * 4);
            memcpy(x, d, (size_t)cur_C * cur_L * 4);
            for (int di = 0; di < 3; di++) {
                int dil = rb_dil[di];
                char wn2[256], bn2[256];
                /* save running state for residual (before leaky_relu modifies x) */
                float *xsave = (float *)malloc((size_t)cur_C * cur_L * 4);
                memcpy(xsave, x, (size_t)cur_C * cur_L * 4);
                /* leaky_relu -> conv1(dil) -> leaky_relu -> conv2(K) */
                for (int i2 = 0; i2 < cur_C * cur_L; i2++) if (x[i2] < 0) x[i2] *= 0.1f;
                snprintf(wn2, sizeof(wn2), "dec.resblocks.%d.convs1.%d.weight", rb_idx, di);
                snprintf(bn2, sizeof(bn2), "dec.resblocks.%d.convs1.%d.bias", rb_idx, di);
                float *h = (float *)malloc((size_t)cur_C * cur_L * 4);
                TMO_BEGIN();
                if (gpu_conv1d(g, wn2, bn2, x, cur_C, cur_L, K, dil, h, cur_C) != 0) {
                    free(x); free(xsave); free(h); free(sum); free(har_source); return -1;
                }
                TMO_CHECK("rb_conv1");
                free(x); x = h;
                for (int i2 = 0; i2 < cur_C * cur_L; i2++) if (x[i2] < 0) x[i2] *= 0.1f;
                snprintf(wn2, sizeof(wn2), "dec.resblocks.%d.convs2.%d.weight", rb_idx, di);
                snprintf(bn2, sizeof(bn2), "dec.resblocks.%d.convs2.%d.bias", rb_idx, di);
                float *s = (float *)malloc((size_t)cur_C * cur_L * 4);
                TMO_BEGIN();
                if (gpu_conv1d(g, wn2, bn2, x, cur_C, cur_L, K, 1, s, cur_C) != 0) {
                    free(xsave); free(x); free(s); free(sum); free(har_source); return -1;
                }
                TMO_CHECK("rb_conv2");
                /* residual: x = xsave + s */
                for (int i = 0; i < cur_C * cur_L; i++) s[i] += xsave[i];
                free(xsave); free(x); x = s;
            }
            for (int i = 0; i < cur_C * cur_L; i++) sum[i] += x[i];
            free(x);
        }
        for (int i = 0; i < cur_C * cur_L; i++) sum[i] /= 3.0f;
        { float rms=0; for(int i=0;i<cur_C*cur_L;i++) rms+=sum[i]*sum[i]; fprintf(stderr, "[dec] stage %d resblocks RMS=%.6f\n", u, sqrtf(rms/(cur_C*cur_L))); }
        free(d); d = sum;
    }

    gpu_leaky_relu(g, d, cur_C * cur_L);
    float *raw = (float *)malloc(cur_L * 4);
    TMO_BEGIN();
    if (gpu_conv1d(g, "dec.conv_post.weight", NULL, d, cur_C, cur_L, 7, 1, raw, 1) != 0) {
        free(d); free(raw); free(har_source); return -1;
    }
    TMO_CHECK("conv_post");
    free(d);
    for (int i = 0; i < cur_L; i++) out[i] = tanhf(raw[i]);
    free(raw);
    free(har_source);
    return 0;
}

/* ---- public API ---- */

int spcv_ggml_flow_dec(const char *gguf_path, int be_id, int gpu_id,
                       const float *m_p, const float *logs_p,
                       const float *g_emb, int T,
                       const float *f0,
                       float *out, int *out_len) {
    fprintf(stderr, "[perf] L449 flow_dec enter be=%d gpu=%d T=%d\n", be_id, gpu_id, T);
    spcv_ggml_t *g = spcv_ggml_load(gguf_path, be_id, gpu_id);
    if (!g) { fprintf(stderr, "[ggml] load fail: %s\n", gguf_path); return -1; }
    fprintf(stderr, "[perf] L452 loaded, starting flow\n");

    float *z = (float *)malloc(FC_DIM * T * 4);
    if (ggml_flow(g, m_p, logs_p, g_emb, T, z) != 0) {
        fprintf(stderr, "[ggml] flow fail\n");
        free(z); spcv_ggml_free(g); return -1;
    }
    fprintf(stderr, "[perf] L458 flow done\n");

    if (getenv("SPCV_GGML_DUMP")) {
        FILE *f = fopen("scratch/ggml_flow_z.bin", "wb");
        if (f) { fwrite(z, sizeof(float), (size_t)FC_DIM * T, f); fclose(f); }
    }

    int OL = T * NUP;
    fprintf(stderr, "[ggml] flow done, starting decoder T=%d OL=%d\n", T, OL);
    if (ggml_decoder(g, z, T, g_emb, f0, out) != 0) {
        fprintf(stderr, "[ggml] decoder fail\n");
        free(z); spcv_ggml_free(g); return -1;
    }
    *out_len = OL;

    free(z);
    spcv_ggml_free(g);
    return 0;
}

int spcv_ggml_infer(const char *gguf_path, int be_id, int gpu_id,
                    const float *emb, int emb_dim, int T,
                    const int64_t *pitch, const float *f0,
                    float *out, int *out_len) {
    (void)emb; (void)emb_dim; (void)pitch; (void)f0;
    /* Placeholder: uses zero m_p/logs_p. Use spcv_ggml_flow_dec() with real encoder output. */
    float *m_p = (float *)calloc(FC_DIM * T, 4);
    float *logs_p = (float *)calloc(FC_DIM * T, 4);
    float *g_emb = (float *)calloc(GIN, 4);
    int rc = spcv_ggml_flow_dec(gguf_path, be_id, gpu_id, m_p, logs_p, g_emb, T, f0, out, out_len);
    free(m_p); free(logs_p); free(g_emb);
    return rc;
}

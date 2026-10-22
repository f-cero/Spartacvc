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
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include <ggml-alloc.h>

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

static float *host_t_ct(const float *in, int C, int L) {
    float *out = (float *)malloc((size_t)C * L * 4);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < L; t++)
            out[t + c * L] = in[c + t * C];
    return out;
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
    struct ggml_tensor *out_t = ggml_transpose(ctx, co);
    ggml_set_name(out_t, "cout");
    spcv_ggml_set_input(g, "xin", xt_host, (size_t)IC * L * 4);
    int out_L = L + 2 * pad - dil * (K - 1);
    if (spcv_ggml_compute(g, out_t, out, (size_t)OC * out_L * 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ggml] conv1d '%s' fail\n", wname);
        free(xt_host); return -1;
    }
    free(xt_host); return 0;
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
    struct ggml_tensor *out_t = ggml_transpose(ctx, co);
    ggml_set_name(out_t, "cout");
    spcv_ggml_set_input(g, "xin", xt_host, (size_t)IC * L * 4);
    int full_L = (L - 1) * stride + K;
    float *full_out = (float *)malloc((size_t)OC * full_L * 4);
    if (spcv_ggml_compute(g, out_t, full_out, (size_t)OC * full_L * 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ggml] conv_t1d '%s' fail\n", wname);
        free(xt_host); free(full_out); return -1;
    }
    int out_L = full_L - 2 * pad;
    for (int c = 0; c < OC; c++)
        for (int t = 0; t < out_L; t++)
            out[c + t * OC] = full_out[c + (t + pad) * OC];
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
    memcpy(z, m_p, (size_t)FC_DIM * T * 4);
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

        snprintf(nm, sizeof(nm), "flow.flows.%d.pre.weight", idx);
        snprintf(bn, sizeof(bn), "flow.flows.%d.pre.bias", idx);
        if (gpu_conv1d(g, nm, bn, z, 96, T, 1, 1, h, FC_DIM) != 0) { free(h); return -1; }

        int CC = 2 * FC_DIM * 3;
        float *cond = (float *)malloc(CC * 4);
        snprintf(nm, sizeof(nm), "flow.flows.%d.enc.cond_layer.weight", idx);
        snprintf(bn, sizeof(bn), "flow.flows.%d.enc.cond_layer.bias", idx);
        if (gpu_conv1d(g, nm, bn, g_emb, GIN, 1, 1, 1, cond, CC) != 0) { free(h); free(cond); return -1; }

        for (int j = 0; j < 3; j++) {
            float *x_in = (float *)malloc((size_t)2 * FC_DIM * T * 4);
            snprintf(nm, sizeof(nm), "flow.flows.%d.enc.in_layers.%d.weight", idx, j);
            snprintf(bn, sizeof(bn), "flow.flows.%d.enc.in_layers.%d.bias", idx, j);
            if (gpu_conv1d(g, nm, bn, h, FC_DIM, T, 5, 1, x_in, 2 * FC_DIM) != 0) { free(h); free(cond); free(x_in); return -1; }
            for (int c = 0; c < FC_DIM; c++) {
                float cv1 = cond[j * 2 * FC_DIM + c];
                float cv2 = cond[j * 2 * FC_DIM + FC_DIM + c];
                for (int t = 0; t < T; t++) {
                    float a = x_in[c * T + t] + cv1;
                    float b = x_in[(FC_DIM+c) * T + t] + cv2;
                    h[c * T + t] = tanhf(a) / (1.0f + expf(-b));
                }
            }
            free(x_in);
            int rs_out = (j < 2) ? 2 * FC_DIM : FC_DIM;
            float *res_skip = (float *)malloc((size_t)rs_out * T * 4);
            snprintf(nm, sizeof(nm), "flow.flows.%d.enc.res_skip_layers.%d.weight", idx, j);
            snprintf(bn, sizeof(bn), "flow.flows.%d.enc.res_skip_layers.%d.bias", idx, j);
            if (gpu_conv1d(g, nm, bn, h, FC_DIM, T, 1, 1, res_skip, rs_out) != 0) { free(h); free(cond); free(res_skip); return -1; }
            for (int c = 0; c < FC_DIM; c++)
                for (int t = 0; t < T; t++)
                    h[c * T + t] += res_skip[c * T + t];
            free(res_skip);
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
                        const float *g_emb, float *out) {
    float *d = (float *)malloc((size_t)UP_DIM * T * 4);
    if (gpu_conv1d(g, "dec.conv_pre.weight", "dec.conv_pre.bias", z, FC_DIM, T, 7, 1, d, UP_DIM) != 0) {
        free(d); return -1;
    }
    /* speaker embedding conditioning */
    {
        float *cond = (float *)malloc(UP_DIM * 4);
        if (gpu_conv1d(g, "dec.cond.weight", "dec.cond.bias", g_emb, GIN, 1, 1, 1, cond, UP_DIM) != 0) {
            free(d); free(cond); return -1;
        }
        for (int t = 0; t < T; t++)
            for (int c = 0; c < UP_DIM; c++)
                d[c * T + t] += cond[c];
        free(cond);
    }

    int cur_C = UP_DIM, cur_L = T;
    for (int u = 0; u < NU; u++) {
        gpu_leaky_relu(g, d, cur_C * cur_L);
        float *d2 = (float *)malloc((size_t)up_OC[u] * cur_L * up_stride[u] * 4);
        char wn[256], bn[256];
        snprintf(wn, sizeof(wn), "dec.ups.%d.weight", u);
        snprintf(bn, sizeof(bn), "dec.ups.%d.bias", u);
        if (gpu_conv_t1d(g, wn, bn, d, cur_C, cur_L, up_K[u], up_stride[u], up_pad[u], d2, up_OC[u]) != 0) {
            free(d); free(d2); return -1;
        }
        free(d); d = d2;
        cur_C = up_OC[u]; cur_L *= up_stride[u];

        float *sum = (float *)calloc((size_t)cur_C * cur_L, 4);
        for (int r = 0; r < 3; r++) {
            int rb_idx = u * 3 + r;
            float *d3 = (float *)malloc((size_t)cur_C * cur_L * 4);
            memcpy(d3, d, (size_t)cur_C * cur_L * 4);
            char wn2[256], bn2[256];
            snprintf(wn2, sizeof(wn2), "dec.resblocks.%d.convs1.%d.weight", rb_idx, r);
            snprintf(bn2, sizeof(bn2), "dec.resblocks.%d.convs1.%d.bias", rb_idx, r);
            if (gpu_conv1d(g, wn2, bn2, d3, cur_C, cur_L, rb_K[r], rb_dil[r], d3, cur_C) != 0) { free(d); free(sum); free(d3); return -1; }
            gpu_leaky_relu(g, d3, cur_C * cur_L);
            snprintf(wn2, sizeof(wn2), "dec.resblocks.%d.convs2.%d.weight", rb_idx, r);
            snprintf(bn2, sizeof(bn2), "dec.resblocks.%d.convs2.%d.bias", rb_idx, r);
            if (gpu_conv1d(g, wn2, bn2, d3, cur_C, cur_L, rb_K[r], 1, d3, cur_C) != 0) { free(d); free(sum); free(d3); return -1; }
            gpu_leaky_relu(g, d3, cur_C * cur_L);
            for (int i = 0; i < cur_C * cur_L; i++) sum[i] += d3[i];
            free(d3);
        }
        for (int i = 0; i < cur_C * cur_L; i++) sum[i] /= 3.0f;
        free(d); d = sum;
    }

    gpu_leaky_relu(g, d, cur_C * cur_L);
    float *raw = (float *)malloc(cur_L * 4);
    if (gpu_conv1d(g, "dec.conv_post.weight", NULL, d, cur_C, cur_L, 7, 1, raw, 1) != 0) {
        free(d); free(raw); return -1;
    }
    free(d);
    for (int i = 0; i < cur_L; i++) out[i] = tanhf(raw[i]);
    free(raw);
    return 0;
}

/* ---- public API ---- */

int spcv_ggml_infer(const char *gguf_path, int be_id, int gpu_id,
                    const float *emb, int emb_dim, int T,
                    const int64_t *pitch, const float *f0,
                    float *out, int *out_len) {
    (void)emb; (void)emb_dim; (void)pitch; (void)f0;

    spcv_ggml_t *g = spcv_ggml_load(gguf_path, be_id, gpu_id);
    if (!g) { fprintf(stderr, "[ggml] load fail: %s\n", gguf_path); return -1; }

    /* Get speaker embedding from GGUF (dec.emb_g or similar) */
    float *g_emb = (float *)calloc(GIN, 4);
    struct ggml_tensor *eg = spcv_ggml_tensor(g, "dec.emb_g");
    if (eg && eg->data) {
        /* On CPU backend we can read directly; on GPU we need to copy */
        if (be_id == 0) {
            memcpy(g_emb, eg->data, GIN * 4);
        }
        /* TODO: GPU copy via ggml_backend_tensor_get */
    }

    /* For now: use m_p from encoder (TBD) — placeholder zeros */
    float *m_p = (float *)calloc(FC_DIM * T, 4);
    float *logs_p = (float *)calloc(FC_DIM * T, 4);
    float *z = (float *)malloc(FC_DIM * T * 4);

    if (ggml_flow(g, m_p, logs_p, g_emb, T, z) != 0) {
        fprintf(stderr, "[ggml] flow fail\n");
        free(g_emb); free(m_p); free(logs_p); free(z);
        spcv_ggml_free(g); return -1;
    }

    int OL = T * NUP;
    if (ggml_decoder(g, z, T, g_emb, out) != 0) {
        fprintf(stderr, "[ggml] decoder fail\n");
        free(g_emb); free(m_p); free(logs_p); free(z);
        spcv_ggml_free(g); return -1;
    }
    *out_len = OL;

    free(g_emb); free(m_p); free(logs_p); free(z);
    spcv_ggml_free(g);
    return 0;
}

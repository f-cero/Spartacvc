/**
 * @file    ggml_rvc_enc_gpu.c
 * @brief   RVC v2 encoder via ggml GPU ops.
 *
 * Structure (6 transformer layers):
 *   1. Embedding: phone one-hot * W_emb + pitch_emb → scale → leaky_relu
 *   2. Per layer:
 *      a. Q/K/V projections (ggml_mul_mat on GPU)
 *      b. Attention with relative positions (host-side)
 *      c. Output projection (ggml_mul_mat on GPU)
 *      d. Residual + LayerNorm + FFN(conv_1d K=3) + LayerNorm (GPU)
 *   3. Final projection → m_p, logs_p
 *
 * Layout: internal tensors are [C, T] (ne[0]=C fastest, C-fast).
 *         Output m_p/logs_p are [FC, T] C-fast.
 */
#include "ggml_rvc_enc_gpu.h"
#include "gguf_load.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ENC_P_DIM   192
#define ENC_FF_DIM  768
#define ENC_HEADS   2
#define ENC_LAYERS  6
#define ENC_FK      3
#define ENC_FC      192
#define ENC_MAX_REL 21
#define ENC_HD      96

/**
 * @brief  Read a tensor's data from GGUF into a host buffer (CPU-side).
 *         Used for relative position embeddings (small, only needed on host).
 */
static float *read_gguf_tensor(const char *gguf_path, const char *name, size_t *out_bytes) {
    struct ggml_context *cctx = NULL;
    struct gguf_init_params cp = { false, &cctx };
    struct gguf_context *cgg = gguf_init_from_file(gguf_path, cp);
    if (!cgg) return NULL;
    struct ggml_tensor *t = ggml_get_tensor(cctx, name);
    if (!t) { gguf_free(cgg); if (cctx) ggml_free(cctx); return NULL; }
    size_t nbytes = ggml_nbytes(t);
    float *d = (float *)malloc(nbytes);
    if (!d) { gguf_free(cgg); if (cctx) ggml_free(cctx); return NULL; }
    memcpy(d, t->data, nbytes);
    gguf_free(cgg);
    if (cctx) ggml_free(cctx);
    if (out_bytes) *out_bytes = nbytes;
    return d;
}

/**
 * @brief  Host-side attention for one head with relative positions.
 *
 * @param  ao       [out] output [hd, T] T-fast: ao[d*T + l1]
 * @param  Q        [hd, T] T-fast: Q[d*T + l1]
 * @param  K        [hd, T] T-fast: K[d*T + l2]
 * @param  V        [hd, T] T-fast: V[d*T + l2]
 * @param  relk     [hd, max_rel] hd-fast: relk[d + idx*hd] (may be NULL)
 * @param  relv     [hd, max_rel] hd-fast: relv[d + idx*hd] (may be NULL)
 * @param  hd       head dimension (96)
 * @param  T        sequence length
 * @param  max_rel  max relative distance (21)
 */
static void attn_head(float *ao, const float *Q, const float *K, const float *V,
                      const float *relk, const float *relv,
                      int hd, int T, int max_rel) {
    int W = (max_rel - 1) / 2;
    float scale = 1.0f / sqrtf((float)hd);
    float *logits = (float *)malloc(T * sizeof(float));
    if (!logits) return;
    for (int l1 = 0; l1 < T; l1++) {
        float mx = -1e9f;
        for (int l2 = 0; l2 < T; l2++) {
            float s = 0;
            for (int d = 0; d < hd; d++) s += Q[d * T + l1] * K[d * T + l2];
            logits[l2] = s * scale;
            if (logits[l2] > mx) mx = logits[l2];
        }
        if (relk)
            for (int l2 = 0; l2 < T; l2++) {
                int idx = W + (l2 - l1);
                if (idx >= 0 && idx < max_rel) {
                    float s = 0;
                    for (int d = 0; d < hd; d++) s += Q[d * T + l1] * relk[d + idx * hd];
                    logits[l2] += s * scale;
                }
            }
        float sum = 0;
        for (int l2 = 0; l2 < T; l2++) { logits[l2] = expf(logits[l2] - mx); sum += logits[l2]; }
        for (int l2 = 0; l2 < T; l2++) logits[l2] /= sum;
        for (int d = 0; d < hd; d++) {
            float acc = 0;
            for (int l2 = 0; l2 < T; l2++) {
                acc += logits[l2] * V[d * T + l2];
                if (relv) {
                    int idx = W + (l2 - l1);
                    if (idx >= 0 && idx < max_rel) acc += logits[l2] * relv[d + idx * hd];
                }
            }
            ao[d * T + l1] = acc;
        }
    }
    free(logits);
}

int spcv_ggml_encoder(const char *gguf_path, int be_id, int gpu_id,
                      const float *phone, int emb_dim, int T,
                      const int64_t *pitch,
                      float *m_p, float *logs_p) {
    (void)gpu_id;
    spcv_ggml_t *g = spcv_ggml_load(gguf_path, be_id, gpu_id);
    if (!g) { fprintf(stderr, "[enc_gpu] gguf load failed\n"); return -1; }
    const char *gp = spcv_ggml_gguf_path(g);
    char nm[192];
    int ok = 0;

    float *x = (float *)malloc((size_t)ENC_P_DIM * T * 4);
    if (!x) goto done;

    /* === Step 1: Embedding (phone + pitch → scale → leaky_relu) === */
    {
        /* Pitch embedding: manual host-side lookup (avoids ggml_get_rows 3D issue) */
        float *pe_host = (float *)malloc((size_t)ENC_P_DIM * T * 4);
        if (!pe_host) goto done;
        {
            int64_t nn;
            size_t pw_bytes = (size_t)ENC_P_DIM * 256 * 4;
            float *pw_data = (float *)malloc(pw_bytes);
            if (!pw_data) { free(pe_host); goto done; }
            if (spcv_ggml_get_tensor(g, "enc_p.emb_pitch.weight", pw_data, pw_bytes, &nn) != 0) {
                fprintf(stderr, "[enc_gpu] emb_pitch.weight missing\n"); free(pw_data); free(pe_host); goto done;
            }
            /* pw is [192, 256, 1]: pw[c + p*192] for channel c, pitch class p */
            for (int t = 0; t < T; t++) {
                int pv = (int)pitch[t];
                if (pv < 0) pv = 0;
                if (pv > 255) pv = 255;
                for (int c = 0; c < ENC_P_DIM; c++)
                    pe_host[c + t * ENC_P_DIM] = pw_data[c + pv * ENC_P_DIM];
            }
            free(pw_data);
        }

        struct ggml_context *ctx = spcv_ggml_ctx(g);
        struct ggml_tensor *phone_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, emb_dim, T);
        ggml_set_name(phone_t, "phone_in");
        struct ggml_tensor *W = spcv_ggml_tensor(g, "enc_p.emb_phone.weight");
        if (!W) { fprintf(stderr, "[enc_gpu] emb_phone.weight missing\n"); free(pe_host); goto done; }
        struct ggml_tensor *x1 = ggml_mul_mat(ctx, W, phone_t); // [P, T]
        struct ggml_tensor *bias = spcv_ggml_tensor(g, "enc_p.emb_phone.bias");
        struct ggml_tensor *x2 = bias ? ggml_add(ctx, x1, bias) : x1;

        struct ggml_tensor *pe_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
        ggml_set_name(pe_t, "pe_in");
        struct ggml_tensor *x3 = ggml_add(ctx, x2, pe_t);
        struct ggml_tensor *x4 = ggml_scale(ctx, x3, sqrtf((float)ENC_P_DIM));
        struct ggml_tensor *x5 = ggml_leaky_relu(ctx, x4, 0.1f, false);
        ggml_set_name(x5, "x_emb");

        spcv_ggml_set_input(g, "phone_in", (void *)phone, (size_t)emb_dim * T * 4);
        spcv_ggml_set_input(g, "pe_in", pe_host, (size_t)ENC_P_DIM * T * 4);
        free(pe_host);
        if (spcv_ggml_compute(g, x5, x, (size_t)ENC_P_DIM * T * 4) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[enc_gpu] embedding compute FAIL\n"); goto done;
        }
    }

    /* Pre-read relative position embeddings (small, used on host) */
    float *relk_buf[ENC_LAYERS], *relv_buf[ENC_LAYERS];
    {
        int64_t nn;
        size_t rel_bytes = (size_t)ENC_MAX_REL * ENC_HD * 4;
        for (int i = 0; i < ENC_LAYERS; i++) {
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.emb_rel_k", i);
            relk_buf[i] = (float *)malloc(rel_bytes);
            if (!relk_buf[i] || spcv_ggml_get_tensor(g, nm, relk_buf[i], rel_bytes, &nn) != 0)
                { fprintf(stderr, "[enc_gpu] rel_k.%d missing\n", i); relk_buf[i] = NULL; }
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.emb_rel_v", i);
            relv_buf[i] = (float *)malloc(rel_bytes);
            if (!relv_buf[i] || spcv_ggml_get_tensor(g, nm, relv_buf[i], rel_bytes, &nn) != 0)
                { fprintf(stderr, "[enc_gpu] rel_v.%d missing\n", i); relv_buf[i] = NULL; }
        }
    }

    /* === Steps 2-7: 6 encoder layers === */
    for (int layer = 0; layer < ENC_LAYERS; layer++) {
        /* --- Q projection (GPU) --- */
        float *Q = (float *)malloc((size_t)ENC_P_DIM * T * 4);
        float *K = (float *)malloc((size_t)ENC_P_DIM * T * 4);
        float *V = (float *)malloc((size_t)ENC_P_DIM * T * 4);
        if (!Q || !K || !V) { free(Q); free(K); free(V); goto done; }

        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.weight", layer);
            struct ggml_tensor *Wq = spcv_ggml_tensor(g, nm);
            if (!Wq) { fprintf(stderr, "[enc_gpu] conv_q.%d missing\n", layer); free(Q);free(K);free(V); goto done; }
            struct ggml_tensor *Wq2d = ggml_reshape_2d(ctx, Wq, ENC_P_DIM, ENC_P_DIM);
            struct ggml_tensor *q1 = ggml_mul_mat(ctx, Wq2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.bias", layer);
            struct ggml_tensor *bq = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *q2 = bq ? ggml_add(ctx, q1, bq) : q1;
            ggml_set_name(q2, "q_out");
            spcv_ggml_set_input(g, "xin", x, (size_t)ENC_P_DIM * T * 4);
            if (spcv_ggml_compute(g, q2, Q, (size_t)ENC_P_DIM * T * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[enc_gpu] Q proj layer %d FAIL\n", layer); free(Q);free(K);free(V); goto done;
            }
        }
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_k.weight", layer);
            struct ggml_tensor *Wk = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *Wk2d = ggml_reshape_2d(ctx, Wk, ENC_P_DIM, ENC_P_DIM);
            struct ggml_tensor *k1 = ggml_mul_mat(ctx, Wk2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_k.bias", layer);
            struct ggml_tensor *bk = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *k2 = bk ? ggml_add(ctx, k1, bk) : k1;
            ggml_set_name(k2, "k_out");
            spcv_ggml_set_input(g, "xin", x, (size_t)ENC_P_DIM * T * 4);
            if (spcv_ggml_compute(g, k2, K, (size_t)ENC_P_DIM * T * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[enc_gpu] K proj layer %d FAIL\n", layer); free(Q);free(K);free(V); goto done;
            }
        }
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_v.weight", layer);
            struct ggml_tensor *Wv = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *Wv2d = ggml_reshape_2d(ctx, Wv, ENC_P_DIM, ENC_P_DIM);
            struct ggml_tensor *v1 = ggml_mul_mat(ctx, Wv2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_v.bias", layer);
            struct ggml_tensor *bv = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *v2 = bv ? ggml_add(ctx, v1, bv) : v1;
            ggml_set_name(v2, "v_out");
            spcv_ggml_set_input(g, "xin", x, (size_t)ENC_P_DIM * T * 4);
            if (spcv_ggml_compute(g, v2, V, (size_t)ENC_P_DIM * T * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[enc_gpu] V proj layer %d FAIL\n", layer); free(Q);free(K);free(V); goto done;
            }
        }

        /* --- Attention on host (relative positions) --- */
        float *ao = (float *)malloc((size_t)ENC_P_DIM * T * 4);
        if (!ao) { free(Q);free(K);free(V); goto done; }
        {
            /* Transpose Q,K,V from C-fast [c+t*P] to T-fast [c*T+t] for head extraction */
            float *Qt = (float *)malloc((size_t)ENC_P_DIM * T * 4);
            float *Kt = (float *)malloc((size_t)ENC_P_DIM * T * 4);
            float *Vt = (float *)malloc((size_t)ENC_P_DIM * T * 4);
            float *aot = (float *)malloc((size_t)ENC_P_DIM * T * 4);
            for (int c = 0; c < ENC_P_DIM; c++)
                for (int t = 0; t < T; t++) {
                    Qt[c * T + t] = Q[c + t * ENC_P_DIM];
                    Kt[c * T + t] = K[c + t * ENC_P_DIM];
                    Vt[c * T + t] = V[c + t * ENC_P_DIM];
                }
            for (int h = 0; h < ENC_HEADS; h++) {
                int hq = h * ENC_HD;
                attn_head(aot + (size_t)hq * T, Qt + (size_t)hq * T,
                          Kt + (size_t)hq * T, Vt + (size_t)hq * T,
                          relk_buf[layer], relv_buf[layer],
                          ENC_HD, T, ENC_MAX_REL);
            }
            /* Transpose ao back to C-fast */
            for (int c = 0; c < ENC_P_DIM; c++)
                for (int t = 0; t < T; t++)
                    ao[c + t * ENC_P_DIM] = aot[c * T + t];
            free(Qt); free(Kt); free(Vt); free(aot);
        }
        free(Q); free(K); free(V);

        /* --- Output projection (GPU) --- */
        float *attn_out = (float *)malloc((size_t)ENC_P_DIM * T * 4);
        if (!attn_out) { free(ao); goto done; }
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_o.weight", layer);
            struct ggml_tensor *Wo = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *Wo2d = ggml_reshape_2d(ctx, Wo, ENC_P_DIM, ENC_P_DIM);
            struct ggml_tensor *o1 = ggml_mul_mat(ctx, Wo2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_o.bias", layer);
            struct ggml_tensor *bo = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *o2 = bo ? ggml_add(ctx, o1, bo) : o1;
            ggml_set_name(o2, "o_out");
            spcv_ggml_set_input(g, "xin", ao, (size_t)ENC_P_DIM * T * 4);
            if (spcv_ggml_compute(g, o2, attn_out, (size_t)ENC_P_DIM * T * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[enc_gpu] O proj layer %d FAIL\n", layer);
                free(ao); free(attn_out); goto done;
            }
        }
        free(ao);

        /* --- Residual + LN + FFN + LN (GPU) --- */
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
            ggml_set_name(xin, "xin");
            struct ggml_tensor *ain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
            ggml_set_name(ain, "ain");

            struct ggml_tensor *res1 = ggml_add(ctx, xin, ain);
            struct ggml_tensor *ln1 = ggml_norm(ctx, res1, 1e-5f);
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_1.%d.gamma", layer);
            struct ggml_tensor *g1 = spcv_ggml_tensor(g, nm);
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_1.%d.beta", layer);
            struct ggml_tensor *b1 = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *ln1o = ggml_add(ctx, ggml_mul(ctx, ln1, g1), b1);

            /* FFN: conv_1d(K=3) → ReLU → conv_1d(K=3) */
            struct ggml_tensor *xt = ggml_cont(ctx, ggml_transpose(ctx, ln1o)); // [T, P]
            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_1.weight", layer);
            struct ggml_tensor *Wf1 = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *h1 = ggml_conv_1d(ctx, Wf1, xt, 1, ENC_FK / 2, 1);
            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_1.bias", layer);
            struct ggml_tensor *bf1 = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *bf1r = ggml_reshape_2d(ctx, bf1, 1, ENC_FF_DIM);
            struct ggml_tensor *h1b = ggml_add(ctx, h1, bf1r);
            struct ggml_tensor *h1r = ggml_relu(ctx, h1b);

            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_2.weight", layer);
            struct ggml_tensor *Wf2 = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *h2 = ggml_conv_1d(ctx, Wf2, h1r, 1, ENC_FK / 2, 1);
            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_2.bias", layer);
            struct ggml_tensor *bf2 = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *bf2r = ggml_reshape_2d(ctx, bf2, 1, ENC_P_DIM);
            struct ggml_tensor *h2b = ggml_add(ctx, h2, bf2r);

            struct ggml_tensor *h2t = ggml_transpose(ctx, h2b); // [P, T]
            struct ggml_tensor *res2 = ggml_add(ctx, ln1o, h2t);
            struct ggml_tensor *ln2 = ggml_norm(ctx, res2, 1e-5f);
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_2.%d.gamma", layer);
            struct ggml_tensor *g2 = spcv_ggml_tensor(g, nm);
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_2.%d.beta", layer);
            struct ggml_tensor *b2 = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *ln2o = ggml_add(ctx, ggml_mul(ctx, ln2, g2), b2);
            ggml_set_name(ln2o, "x_out");

            spcv_ggml_set_input(g, "xin", x, (size_t)ENC_P_DIM * T * 4);
            spcv_ggml_set_input(g, "ain", attn_out, (size_t)ENC_P_DIM * T * 4);
            if (spcv_ggml_compute(g, ln2o, x, (size_t)ENC_P_DIM * T * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[enc_gpu] LN+FFN layer %d FAIL\n", layer);
                free(attn_out); goto done;
            }
        }
        free(attn_out);
    }

    /* === Final projection → m_p, logs_p === */
    {
        struct ggml_context *ctx = spcv_ggml_ctx(g);
        struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ENC_P_DIM, T);
        ggml_set_name(xin, "xin");
        struct ggml_tensor *Wp = spcv_ggml_tensor(g, "enc_p.proj.weight");
        if (!Wp) { fprintf(stderr, "[enc_gpu] proj.weight missing\n"); goto done; }
        struct ggml_tensor *Wp2d = ggml_reshape_2d(ctx, Wp, ENC_P_DIM, 2 * ENC_FC);
        struct ggml_tensor *s1 = ggml_mul_mat(ctx, Wp2d, xin); // [2FC, T]
        struct ggml_tensor *bp = spcv_ggml_tensor(g, "enc_p.proj.bias");
        struct ggml_tensor *s2 = bp ? ggml_add(ctx, s1, bp) : s1;
        ggml_set_name(s2, "stats");
        spcv_ggml_set_input(g, "xin", x, (size_t)ENC_P_DIM * T * 4);
        float *stats = (float *)malloc((size_t)2 * ENC_FC * T * 4);
        if (spcv_ggml_compute(g, s2, stats, (size_t)2 * ENC_FC * T * 4) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[enc_gpu] proj FAIL\n"); free(stats); goto done;
        }
        /* stats is [2FC, T] C-fast: stats[c + t*2FC]
         * Output must be T-fast: m_p[c*T + t] to match pure-C / flow expectation */
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < ENC_FC; c++) {
                m_p[c * T + t] = stats[c + t * 2 * ENC_FC];
                logs_p[c * T + t] = stats[ENC_FC + c + t * 2 * ENC_FC];
            }
        }
        free(stats);
    }

    ok = 1;
    if (getenv("SPCV_DBG_ENC")) {
        FILE *f = fopen("scratch/ggml_enc_m_p.bin", "wb");
        fwrite(m_p, 4, (size_t)ENC_FC * T, f); fclose(f);
        f = fopen("scratch/ggml_enc_logs_p.bin", "wb");
        fwrite(logs_p, 4, (size_t)ENC_FC * T, f); fclose(f);
        fprintf(stderr, "[enc_gpu] dumped m_p/logs_p (%d x %d)\n", ENC_FC, T);
    }
done:
    for (int i = 0; i < ENC_LAYERS; i++) { free(relk_buf[i]); free(relv_buf[i]); }
    free(x);
    spcv_ggml_free(g);
    return ok ? 0 : -1;
}

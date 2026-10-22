// test_rvc_enc_gpu.c — Full RVC v2 encoder using ggml GPU ops.
//
// Strategy: most compute on GPU (mul_mat, conv_1d, norm, leaky_relu).
// Attention (relative positions) done on host between GPU steps.
// Multiple compute calls: embedding → [QKV → host-attn → post] × 6 → proj.
//
// Layout (ggml): ne[0] fastest.
//   Data: [C, T] → ne[0]=C, ne[1]=T.  data[c + t*C]
//   Weight (linear): [Cin, Cout] → ne[0]=Cin, ne[1]=Cout.
//   mul_mat(W [Cin,Cout], x [Cin,T]) → [Cout, T]
//   conv_1d(W [K,IC,OC], x [T,IC]) → [T, OC]  (note: T is ne[0])
//   norm: normalizes over ne[0]

#include "gguf_load.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "rvc.h"
#include "rvc_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define P_DIM 192
#define Q_DIM 768
#define FF_DIM 768
#define N_HEADS 2
#define N_LAYERS 6
#define FK 3
#define FC 192
#define MAX_REL 21
#define HD 96
#define T_LEN 16

static double maxabs_err(const float *a, const float *b, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > m) m = d; }
    return m;
}

// Read tensor data from GGUF on CPU (for host-side attention + reference).
static float *cpu_tensor(const char *gguf_path, const char *name) {
    struct ggml_context *cctx = NULL;
    struct gguf_init_params cp = { false, &cctx };
    struct gguf_context *cgg = gguf_init_from_file(gguf_path, cp);
    if (!cgg) return NULL;
    struct ggml_tensor *t = ggml_get_tensor(cctx, name);
    if (!t) { gguf_free(cgg); if (cctx) ggml_free(cctx); return NULL; }
    size_t nbytes = ggml_nbytes(t);
    float *d = (float *)malloc(nbytes);
    memcpy(d, t->data, nbytes);
    gguf_free(cgg);
    if (cctx) ggml_free(cctx);
    return d;
}

// Host-side attention for one head.
// Layout: T-fast — Q[d*T + l1] where d in [0,hd), l1 in [0,T).
// relk/relv: [hd, max_rel] with hd fast: relk[d + idx*hd].
static void attn_head(float *ao, const float *Q, const float *K, const float *V,
                      const float *relk, const float *relv,
                      int hd, int T, int max_rel) {
    int W = (max_rel - 1) / 2;
    float scale = 1.0f / sqrtf((float)hd);
    for (int l1 = 0; l1 < T; l1++) {
        float mx = -1e9f;
        float logits[1024];
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
}

int main(int argc, char **argv) {
    int beid = (argc > 1 && !strcmp(argv[1], "cuda")) ? SPCV_BE_CUDA : SPCV_BE_CPU;
    const char *gguf_path = "scratch/nahida.gguf";

    spcv_ggml_t *g = spcv_ggml_load(gguf_path, beid, 0);
    if (!g) { fprintf(stderr, "load failed\n"); return 2; }
    const char *gp = spcv_ggml_gguf_path(g);

    // Input: random phone [T, Q] and pitch [T]
    float *phone = malloc((size_t)T_LEN * Q_DIM * 4);
    int64_t *pitch = malloc(T_LEN * 8);
    srand(42);
    for (int i = 0; i < T_LEN * Q_DIM; i++) phone[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int t = 0; t < T_LEN; t++) pitch[t] = (t * 37) % 256;

    // ---- GGML GPU encoder (must run BEFORE pure-C to avoid heap corruption) ----
    float *x = malloc((size_t)P_DIM * T_LEN * 4);  // [P, T] ne[0]=P, ne[1]=T

    // === Step 1: Embedding ===
    {
        struct ggml_context *ctx = spcv_ggml_ctx(g);
        struct ggml_tensor *phone_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Q_DIM, T_LEN);
        ggml_set_name(phone_t, "phone_in");
        struct ggml_tensor *W = spcv_ggml_tensor(g, "enc_p.emb_phone.weight"); // [Q, P]
        struct ggml_tensor *x1 = ggml_mul_mat(ctx, W, phone_t); // [P, T]
        struct ggml_tensor *bias = spcv_ggml_tensor(g, "enc_p.emb_phone.bias"); // [P]
        struct ggml_tensor *x2 = ggml_add(ctx, x1, bias); // [P, T] + [P] broadcast
        // Pitch embedding
        int32_t *idx32 = malloc(T_LEN * 4);
        for (int t = 0; t < T_LEN; t++) idx32[t] = (int)pitch[t];
        struct ggml_tensor *idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_LEN);
        ggml_set_name(idx_t, "pitch_idx");
        struct ggml_tensor *pw = spcv_ggml_tensor(g, "enc_p.emb_pitch.weight"); // [P, 256]
        struct ggml_tensor *pe = ggml_get_rows(ctx, pw, idx_t); // [P, T]
        struct ggml_tensor *x3 = ggml_add(ctx, x2, pe); // [P, T]
        // Scale + LeakyReLU
        struct ggml_tensor *x4 = ggml_scale(ctx, x3, sqrtf((float)P_DIM));
        struct ggml_tensor *x5 = ggml_leaky_relu(ctx, x4, 0.1f, false);
        ggml_set_name(x5, "x_emb");

        spcv_ggml_set_input(g, "phone_in", phone, (size_t)Q_DIM * T_LEN * 4);
        spcv_ggml_set_input(g, "pitch_idx", idx32, T_LEN * 4);
        if (spcv_ggml_compute(g, x5, x, (size_t)P_DIM * T_LEN * 4) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "embedding compute FAIL\n"); return 2;
        }
        free(idx32);
    }

    // === Steps 2-7: 6 encoder layers ===
    for (int layer = 0; layer < N_LAYERS; layer++) {
        char nm[128];

        // --- 2a: Q projection (GPU) ---
        float *Q = malloc((size_t)P_DIM * T_LEN * 4);
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.weight", layer);
            struct ggml_tensor *Wq = spcv_ggml_tensor(g, nm); // [1,P,P]
            struct ggml_tensor *Wq2d = ggml_reshape_2d(ctx, Wq, P_DIM, P_DIM);
            struct ggml_tensor *q1 = ggml_mul_mat(ctx, Wq2d, xin); // [P, T]
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.bias", layer);
            struct ggml_tensor *bq = spcv_ggml_tensor(g, nm); // [P]
            struct ggml_tensor *q2 = ggml_add(ctx, q1, bq);
            ggml_set_name(q2, "q_out");
            spcv_ggml_set_input(g, "xin", x, (size_t)P_DIM * T_LEN * 4);
            if (spcv_ggml_compute(g, q2, Q, (size_t)P_DIM * T_LEN * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "Q proj layer %d FAIL\n", layer); return 2;
            }
        }

        // --- 2b: K projection (GPU) ---
        float *K = malloc((size_t)P_DIM * T_LEN * 4);
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_k.weight", layer);
            struct ggml_tensor *Wk = spcv_ggml_tensor(g, nm); // [1,P,P]
            struct ggml_tensor *Wk2d = ggml_reshape_2d(ctx, Wk, P_DIM, P_DIM);
            struct ggml_tensor *k1 = ggml_mul_mat(ctx, Wk2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_k.bias", layer);
            struct ggml_tensor *bk = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *k2 = ggml_add(ctx, k1, bk);
            ggml_set_name(k2, "k_out");
            spcv_ggml_set_input(g, "xin", x, (size_t)P_DIM * T_LEN * 4);
            if (spcv_ggml_compute(g, k2, K, (size_t)P_DIM * T_LEN * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "K proj layer %d FAIL\n", layer); return 2;
            }
        }

        // --- 2c: V projection (GPU) ---
        float *V = malloc((size_t)P_DIM * T_LEN * 4);
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_v.weight", layer);
            struct ggml_tensor *Wv = spcv_ggml_tensor(g, nm); // [1,P,P]
            struct ggml_tensor *Wv2d = ggml_reshape_2d(ctx, Wv, P_DIM, P_DIM);
            struct ggml_tensor *v1 = ggml_mul_mat(ctx, Wv2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_v.bias", layer);
            struct ggml_tensor *bv = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *v2 = ggml_add(ctx, v1, bv);
            ggml_set_name(v2, "v_out");
            spcv_ggml_set_input(g, "xin", x, (size_t)P_DIM * T_LEN * 4);
            if (spcv_ggml_compute(g, v2, V, (size_t)P_DIM * T_LEN * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "V proj layer %d FAIL\n", layer); return 2;
            }
        }

        // --- 2d: Attention on host ---
        // Q,K,V are in C-fast layout: data[c + t*P]. Transpose to T-fast: data[c*T + t].
        // Then head h is at offset h*HD*T (contiguous).
        float *ao = malloc((size_t)P_DIM * T_LEN * 4); // output in C-fast
        {
            // Transpose Q,K,V to T-fast
            float *Qt = malloc((size_t)P_DIM * T_LEN * 4);
            float *Kt = malloc((size_t)P_DIM * T_LEN * 4);
            float *Vt = malloc((size_t)P_DIM * T_LEN * 4);
            float *aot = malloc((size_t)P_DIM * T_LEN * 4);
            for (int c = 0; c < P_DIM; c++)
                for (int t = 0; t < T_LEN; t++) {
                    Qt[c * T_LEN + t] = Q[c + t * P_DIM];
                    Kt[c * T_LEN + t] = K[c + t * P_DIM];
                    Vt[c * T_LEN + t] = V[c + t * P_DIM];
                }
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.emb_rel_k", layer);
            float *relk = cpu_tensor(gp, nm);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.emb_rel_v", layer);
            float *relv = cpu_tensor(gp, nm);
            for (int h = 0; h < N_HEADS; h++) {
                int hq = h * HD;
                attn_head(aot + (size_t)hq * T_LEN, Qt + (size_t)hq * T_LEN,
                          Kt + (size_t)hq * T_LEN, Vt + (size_t)hq * T_LEN,
                          relk, relv, HD, T_LEN, MAX_REL);
            }
            free(relk); free(relv);
            // Transpose ao back to C-fast
            for (int c = 0; c < P_DIM; c++)
                for (int t = 0; t < T_LEN; t++)
                    ao[c + t * P_DIM] = aot[c * T_LEN + t];
            free(Qt); free(Kt); free(Vt); free(aot);
        }
        free(Q); free(K); free(V);

        // --- 2e: Output projection (GPU) ---
        float *attn_out = malloc((size_t)P_DIM * T_LEN * 4);
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
            ggml_set_name(xin, "xin");
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_o.weight", layer);
            struct ggml_tensor *Wo = spcv_ggml_tensor(g, nm); // [1,P,P]
            struct ggml_tensor *Wo2d = ggml_reshape_2d(ctx, Wo, P_DIM, P_DIM);
            struct ggml_tensor *o1 = ggml_mul_mat(ctx, Wo2d, xin);
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_o.bias", layer);
            struct ggml_tensor *bo = spcv_ggml_tensor(g, nm);
            struct ggml_tensor *o2 = ggml_add(ctx, o1, bo);
            ggml_set_name(o2, "o_out");
            spcv_ggml_set_input(g, "xin", ao, (size_t)P_DIM * T_LEN * 4);
            if (spcv_ggml_compute(g, o2, attn_out, (size_t)P_DIM * T_LEN * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "O proj layer %d FAIL\n", layer); return 2;
            }
        }
        free(ao);

        // --- 2f: Residual + LN + FFN + LN (GPU) ---
        {
            struct ggml_context *ctx = spcv_ggml_ctx(g);
            struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
            ggml_set_name(xin, "xin");
            struct ggml_tensor *ain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
            ggml_set_name(ain, "ain");

            // Residual: x + attn_out
            struct ggml_tensor *res1 = ggml_add(ctx, xin, ain); // [P, T]

            // LayerNorm 1: norm over ne[0]=P, then * gamma + beta
            struct ggml_tensor *ln1 = ggml_norm(ctx, res1, 1e-5f); // [P, T] normalized
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_1.%d.gamma", layer);
            struct ggml_tensor *g1 = spcv_ggml_tensor(g, nm); // [P]
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_1.%d.beta", layer);
            struct ggml_tensor *b1 = spcv_ggml_tensor(g, nm); // [P]
            struct ggml_tensor *ln1o = ggml_add(ctx, ggml_mul(ctx, ln1, g1), b1); // [P, T]

            // FFN: conv_1d(K=3) → ReLU → conv_1d(K=3)
            // conv_1d expects data [T, IC] (ne[0]=T). Our data is [P, T] (ne[0]=P).
            // Need transpose: [P, T] → [T, P]
            struct ggml_tensor *xt = ggml_cont(ctx, ggml_transpose(ctx, ln1o)); // [T, P] contiguous
            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_1.weight", layer);
            struct ggml_tensor *Wf1 = spcv_ggml_tensor(g, nm); // [3, P, FF]
            struct ggml_tensor *h1 = ggml_conv_1d(ctx, Wf1, xt, 1, FK / 2, 1); // [T, FF]
            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_1.bias", layer);
            struct ggml_tensor *bf1 = spcv_ggml_tensor(g, nm); // [FF]
            struct ggml_tensor *bf1r = ggml_reshape_2d(ctx, bf1, 1, FF_DIM); // [1, FF]
            struct ggml_tensor *h1b = ggml_add(ctx, h1, bf1r); // [T, FF] + [1, FF]
            struct ggml_tensor *h1r = ggml_relu(ctx, h1b); // [T, FF]

            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_2.weight", layer);
            struct ggml_tensor *Wf2 = spcv_ggml_tensor(g, nm); // [3, FF, P]
            struct ggml_tensor *h2 = ggml_conv_1d(ctx, Wf2, h1r, 1, FK / 2, 1); // [T, P]
            snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_2.bias", layer);
            struct ggml_tensor *bf2 = spcv_ggml_tensor(g, nm); // [P]
            struct ggml_tensor *bf2r = ggml_reshape_2d(ctx, bf2, 1, P_DIM); // [1, P]
            struct ggml_tensor *h2b = ggml_add(ctx, h2, bf2r); // [T, P] + [1, P]

            // Transpose back: [T, P] → [P, T]
            struct ggml_tensor *h2t = ggml_transpose(ctx, h2b); // [P, T]

            // Residual + LayerNorm 2
            struct ggml_tensor *res2 = ggml_add(ctx, ln1o, h2t); // [P, T]
            struct ggml_tensor *ln2 = ggml_norm(ctx, res2, 1e-5f); // [P, T]
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_2.%d.gamma", layer);
            struct ggml_tensor *g2 = spcv_ggml_tensor(g, nm); // [P]
            snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_2.%d.beta", layer);
            struct ggml_tensor *b2 = spcv_ggml_tensor(g, nm); // [P]
            struct ggml_tensor *ln2o = ggml_add(ctx, ggml_mul(ctx, ln2, g2), b2); // [P, T]
            ggml_set_name(ln2o, "x_out");

            spcv_ggml_set_input(g, "xin", x, (size_t)P_DIM * T_LEN * 4);
            spcv_ggml_set_input(g, "ain", attn_out, (size_t)P_DIM * T_LEN * 4);
            if (spcv_ggml_compute(g, ln2o, x, (size_t)P_DIM * T_LEN * 4) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "LN+FFN layer %d FAIL\n", layer); return 2;
            }
        }
        free(attn_out);
    }

    // === Step 8: Final projection ===
    float *m_p = malloc((size_t)FC * T_LEN * 4);
    float *logs_p = malloc((size_t)FC * T_LEN * 4);
    {
        struct ggml_context *ctx = spcv_ggml_ctx(g);
        struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P_DIM, T_LEN);
        ggml_set_name(xin, "xin");
        struct ggml_tensor *Wp = spcv_ggml_tensor(g, "enc_p.proj.weight"); // [1, P, 2FC]
        struct ggml_tensor *Wp2d = ggml_reshape_2d(ctx, Wp, P_DIM, 2 * FC);
        struct ggml_tensor *s1 = ggml_mul_mat(ctx, Wp2d, xin); // [2FC, T]
        struct ggml_tensor *bp = spcv_ggml_tensor(g, "enc_p.proj.bias"); // [2FC]
        struct ggml_tensor *s2 = ggml_add(ctx, s1, bp); // [2FC, T]
        ggml_set_name(s2, "stats");
        spcv_ggml_set_input(g, "xin", x, (size_t)P_DIM * T_LEN * 4);
        float *stats = malloc((size_t)2 * FC * T_LEN * 4);
        if (spcv_ggml_compute(g, s2, stats, (size_t)2 * FC * T_LEN * 4) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "proj FAIL\n"); return 2;
        }
        // stats is [2FC, T] C-fast: stats[c + t*2FC] for c in [0,2FC)
        // m_p = channels 0..FC-1, logs_p = channels FC..2FC-1
        for (int t = 0; t < T_LEN; t++) {
            for (int c = 0; c < FC; c++) {
                m_p[c + t * FC] = stats[c + t * 2 * FC];
                logs_p[c + t * FC] = stats[FC + c + t * 2 * FC];
            }
        }
        free(stats);
    }

    // === Save output to files for external comparison ===
    {
        FILE *f = fopen("scratch/enc_gpu_m_p.bin", "wb");
        fwrite(m_p, sizeof(float), FC * T_LEN, f);
        fclose(f);
        f = fopen("scratch/enc_gpu_logs_p.bin", "wb");
        fwrite(logs_p, sizeof(float), FC * T_LEN, f);
        fclose(f);
        printf("saved scratch/enc_gpu_m_p.bin (%d floats)\n", FC * T_LEN);
        printf("m_p[0..5]=%g %g %g %g %g %g\n", m_p[0],m_p[1],m_p[2],m_p[3],m_p[4],m_p[5]);
        printf("logs_p[0..5]=%g %g %g %g %g %g\n", logs_p[0],logs_p[1],logs_p[2],logs_p[3],logs_p[4],logs_p[5]);
    }

    free(x); free(phone); free(pitch);
    free(m_p); free(logs_p);
    spcv_ggml_free(g);
    printf("DONE\n");
    return 0;
}

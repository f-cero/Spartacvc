// ============================================================================
// RVC v2 Flow — ggml GPU compute test
//
// Flow structure (reverse order):
//   VAE: z = m_p + exp(logs_p) * randn * 0.66666
//   Flip channels
//   4 blocks (GGUF idx: 6, 4, 2, 0):
//     pre:   Conv1d(HP→FC, K=1)
//     cond:  Conv1d(GIN→2*FC*NL, K=1)
//     WaveNet (3 layers):
//       in:    Conv1d(FC→2FC, K=5)
//       act:   tanh(x[:FC]+c1) * sigmoid(x[FC:]+c2)
//       res:   Conv1d(FC→2FC|FC, K=1)
//     post:  Conv1d(FC→HP, K=1)
//     x1 -= m
//     Flip (if not last)
//
// GPU: all conv_1d ops
// Host: VAE, flip, activation, residual, coupling
// ============================================================================
#include "gguf_load.h"
#include "rvc.h"
#include "rvc_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>
#include <ggml-alloc.h>

#define FC_DIM  192
#define HP_DIM  96
#define GIN_DIM 256
#define NL      3   // wavenet layers
#define T_LEN   16  // sequence length (short for testing)

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
    ggml_free(cctx);
    return d;
}

static double maxabs_err(const float *a, const float *b, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - (double)b[i]); if (d > m) m = d; }
    return m;
}

// Reverse channel order: out[(FC-1-c)*L + t] = in[c*L + t]
static void flip_channels(float *buf, int FC, int L) {
    float *tmp = (float *)malloc((size_t)FC * L * 4);
    for (int c = 0; c < FC; c++)
        memcpy(&tmp[c * L], &buf[(FC - 1 - c) * L], (size_t)L * 4);
    memcpy(buf, tmp, (size_t)FC * L * 4);
    free(tmp);
}

// GPU conv_1d helper: input [IC, L] C-fast → output [OC, L] C-fast
// Weight in GGUF: [K, IC, OC]
// Transposes input on host to [L, IC] before upload (avoids ggml_cont in graph).
static float *host_transpose_ct(const float *in, int IC, int L) {
    // in: [IC, L] C-fast: in[c + t*IC]
    // out: [L, IC] C-fast: out[t + c*L]
    float *out = (float *)malloc((size_t)IC * L * 4);
    for (int c = 0; c < IC; c++)
        for (int t = 0; t < L; t++)
            out[t + c * L] = in[c + t * IC];
    return out;
}

static int gpu_conv1d(spcv_ggml_t *g, const char *wname, const char *bname,
                      const float *in, int IC, int L, int K,
                      float *out, int OC) {
    struct ggml_tensor *W = spcv_ggml_tensor(g, wname);
    if (!W) { fprintf(stderr, "weight '%s' not found\n", wname); return -1; }

    // Transpose on host: [IC, L] → [L, IC]
    float *xt_host = host_transpose_ct(in, IC, L);

    struct ggml_context *ctx = spcv_ggml_ctx(g);
    // Input: [L, IC] (ne[0]=L, ne[1]=IC) — already in correct layout
    struct ggml_tensor *xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);
    ggml_set_name(xin, "xin");
    // Conv
    int pad = (K - 1) / 2;
    struct ggml_tensor *co = ggml_conv_1d(ctx, W, xin, 1, pad, 1); // [L, OC]
    // Add bias: bias is [OC], need [1, OC] for broadcasting
    if (bname) {
        struct ggml_tensor *bias = spcv_ggml_tensor(g, bname);
        if (bias) {
            struct ggml_tensor *br = ggml_reshape_2d(ctx, bias, 1, OC);
            co = ggml_add(ctx, co, br);
        }
    }
    // Transpose back: [L, OC] → [OC, L] (output in C-fast)
    struct ggml_tensor *out_t = ggml_transpose(ctx, co); // [OC, L]
    ggml_set_name(out_t, "cout");

    spcv_ggml_set_input(g, "xin", xt_host, (size_t)IC * L * 4);
    if (spcv_ggml_compute(g, out_t, out, (size_t)OC * L * 4) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "conv1d compute fail\n");
        free(xt_host);
        return -1;
    }
    free(xt_host);
    return 0;
}

int main(int argc, char **argv) {
    int beid = (argc > 1 && !strcmp(argv[1], "cuda")) ? SPCV_BE_CUDA : SPCV_BE_CPU;
    const char *gguf_path = "scratch/nahida.gguf";

    spcv_ggml_t *g = spcv_ggml_load(gguf_path, beid, 0);
    if (!g) { fprintf(stderr, "load failed\n"); return 2; }
    const char *gp = spcv_ggml_gguf_path(g);

    // Input: m_p [FC, T] C-fast, logs_p [FC, T] C-fast, g_emb [GIN]
    float *m_p = (float *)malloc((size_t)FC_DIM * T_LEN * 4);
    float *logs_p = (float *)malloc((size_t)FC_DIM * T_LEN * 4);
    float *g_emb = (float *)malloc(GIN_DIM * 4);
    srand(42);
    for (int i = 0; i < FC_DIM * T_LEN; i++) {
        m_p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        logs_p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }
    for (int i = 0; i < GIN_DIM; i++) g_emb[i] = ((float)rand() / RAND_MAX - 0.5f);

    // === GGML GPU Flow ===
    float *z = (float *)malloc((size_t)FC_DIM * T_LEN * 4);
    memcpy(z, m_p, (size_t)FC_DIM * T_LEN * 4);
    // VAE: z = m_p + exp(logs_p) * 0 (deterministic, no noise for testing)
    // For deterministic test, use zero noise: z = m_p
    // (In production, z = m_p + exp(logs_p) * randn * 0.66666)

    // Flip channels
    flip_channels(z, FC_DIM, T_LEN);

    // 4 blocks in reverse order: GGUF idx 6, 4, 2, 0
    int gguf_idx[4] = { 6, 4, 2, 0 };
    for (int bi = 0; bi < 4; bi++) {
        int gi = gguf_idx[bi];
        char nm[256];

        // 1. pre: Conv1d(HP→FC, K=1) on first HP channels
        // Input: z[:HP] [HP, L], Output: h [FC, L]
        float *h = (float *)malloc((size_t)FC_DIM * T_LEN * 4);
        {
            // Extract first HP channels (C-fast: z[c + t*FC] for c in [0,HP))
            float *z_hp = (float *)malloc((size_t)HP_DIM * T_LEN * 4);
            for (int t = 0; t < T_LEN; t++)
                for (int c = 0; c < HP_DIM; c++)
                    z_hp[c + t * HP_DIM] = z[c + t * FC_DIM];
            snprintf(nm, sizeof(nm), "flow.flows.%d.pre.weight", gi);
            snprintf(nm + 200, 1, "\0"); // prevent overflow
            char wname[256], bname[256];
            snprintf(wname, sizeof(wname), "flow.flows.%d.pre.weight", gi);
            snprintf(bname, sizeof(bname), "flow.flows.%d.pre.bias", gi);
            if (gpu_conv1d(g, wname, bname, z_hp, HP_DIM, T_LEN, 1, h, FC_DIM) != 0) {
                fprintf(stderr, "block %d pre fail\n", gi); return 2;
            }
            free(z_hp);
        }

        // 2. cond: Conv1d(GIN→2*FC*NL, K=1)
        int CC = 2 * FC_DIM * NL; // 1152
        float *cond = (float *)malloc(CC * 4);
        {
            char wname[256], bname[256];
            snprintf(wname, sizeof(wname), "flow.flows.%d.enc.cond_layer.weight", gi);
            snprintf(bname, sizeof(bname), "flow.flows.%d.enc.cond_layer.bias", gi);
            // g_emb is [GIN] 1D, treat as [GIN, 1] for conv
            if (gpu_conv1d(g, wname, bname, g_emb, GIN_DIM, 1, 1, cond, CC) != 0) {
                fprintf(stderr, "block %d cond fail\n", gi); return 2;
            }
        }

        // 3. WaveNet (3 layers)
        for (int j = 0; j < NL; j++) {
            // in_layer: Conv1d(FC→2FC, K=5)
            float *x_in = (float *)malloc((size_t)2 * FC_DIM * T_LEN * 4);
            {
                char wname[256], bname[256];
                snprintf(wname, sizeof(wname), "flow.flows.%d.enc.in_layers.%d.weight", gi, j);
                snprintf(bname, sizeof(bname), "flow.flows.%d.enc.in_layers.%d.bias", gi, j);
                if (gpu_conv1d(g, wname, bname, h, FC_DIM, T_LEN, 5, x_in, 2 * FC_DIM) != 0) {
                    fprintf(stderr, "block %d wn%d in fail\n", gi, j); return 2;
                }
            }

            // fused_add_tanh_sigmoid_multiply (host)
            // a = x_in[:FC] + cond[j*2FC : j*2FC+FC]
            // b = x_in[FC:2FC] + cond[j*2FC+FC : j*2FC+2FC]
            // out = tanh(a) * sigmoid(b)
            for (int c = 0; c < FC_DIM; c++) {
                float cv1 = cond[j * 2 * FC_DIM + c];
                float cv2 = cond[j * 2 * FC_DIM + FC_DIM + c];
                for (int t = 0; t < T_LEN; t++) {
                    float a = x_in[c + t * 2 * FC_DIM] + cv1;
                    float b = x_in[FC_DIM + c + t * 2 * FC_DIM] + cv2;
                    h[c + t * FC_DIM] = tanhf(a) / (1.0f + expf(-b));
                }
            }
            free(x_in);

            // res_skip: Conv1d(FC→2FC or FC, K=1)
            int rs_out = (j < NL - 1) ? 2 * FC_DIM : FC_DIM;
            float *res_skip = (float *)malloc((size_t)rs_out * T_LEN * 4);
            {
                char wname[256], bname[256];
                snprintf(wname, sizeof(wname), "flow.flows.%d.enc.res_skip_layers.%d.weight", gi, j);
                snprintf(bname, sizeof(bname), "flow.flows.%d.enc.res_skip_layers.%d.bias", gi, j);
                if (gpu_conv1d(g, wname, bname, h, FC_DIM, T_LEN, 1, res_skip, rs_out) != 0) {
                    fprintf(stderr, "block %d wn%d rs fail\n", gi, j); return 2;
                }
            }

            // residual + skip
            if (j < NL - 1) {
                for (int c = 0; c < FC_DIM; c++)
                    for (int t = 0; t < T_LEN; t++) {
                        h[c + t * FC_DIM] += res_skip[c + t * rs_out];
                        // accumulate to output (stored in z temporarily)
                        z[c + t * FC_DIM] += res_skip[FC_DIM + c + t * rs_out];
                    }
            } else {
                for (int c = 0; c < FC_DIM; c++)
                    for (int t = 0; t < T_LEN; t++)
                        z[c + t * FC_DIM] += res_skip[c + t * rs_out];
            }
            free(res_skip);
        }
        // WaveNet output is in z (the skip output)
        // Copy back to h for post
        memcpy(h, z, (size_t)FC_DIM * T_LEN * 4);

        // 4. post: Conv1d(FC→HP, K=1)
        float *m_out = (float *)malloc((size_t)HP_DIM * T_LEN * 4);
        {
            char wname[256], bname[256];
            snprintf(wname, sizeof(wname), "flow.flows.%d.post.weight", gi);
            snprintf(bname, sizeof(bname), "flow.flows.%d.post.bias", gi);
            if (gpu_conv1d(g, wname, bname, h, FC_DIM, T_LEN, 1, m_out, HP_DIM) != 0) {
                fprintf(stderr, "block %d post fail\n", gi); return 2;
            }
        }

        // 5. Reverse coupling: x1 = x1 - m
        // x1 is the second half of z: z[HP:FC]
        for (int c = 0; c < HP_DIM; c++)
            for (int t = 0; t < T_LEN; t++)
                z[HP_DIM + c + t * FC_DIM] -= m_out[c + t * HP_DIM];
        free(m_out);

        // 6. Flip (if not last block)
        if (bi < 3) {
            flip_channels(z, FC_DIM, T_LEN);
        }
        free(h);
        free(cond);
    }

    // Save output
    {
        FILE *f = fopen("scratch/flow_gpu_out.bin", "wb");
        fwrite(z, sizeof(float), FC_DIM * T_LEN, f);
        fclose(f);
        printf("saved scratch/flow_gpu_out.bin (%d floats)\n", FC_DIM * T_LEN);
        printf("z[0..5]=%g %g %g %g %g %g\n", z[0],z[1],z[2],z[3],z[4],z[5]);
    }

    free(z); free(m_p); free(logs_p); free(g_emb);
    spcv_ggml_free(g);
    printf("DONE\n");
    return 0;
}

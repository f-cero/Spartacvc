// Generate pure-C flow reference (zero noise for deterministic comparison)
#include "rvc.h"
#include "rvc_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FC 192
#define HP 96
#define GIN 256
#define NL 3
#define T_LEN 16

int main() {
    rvc_model_t *m = (rvc_model_t *)calloc(1, sizeof(rvc_model_t));
    if (rvc_load(m, "models/voices/nahida-jp.pth") != 0) { fprintf(stderr, "load fail\n"); return 2; }

    // Same input as ggml test
    float *m_p = malloc((size_t)FC * T_LEN * 4);
    float *logs_p = malloc((size_t)FC * T_LEN * 4);
    float *g_emb = malloc(GIN * 4);
    srand(42);
    for (int i = 0; i < FC * T_LEN; i++) {
        m_p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        logs_p[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    }
    for (int i = 0; i < GIN; i++) g_emb[i] = ((float)rand() / RAND_MAX - 0.5f);

    // Flow with zero noise: z = m_p (no reparameterization)
    // This is what the ggml test does
    float *z = malloc((size_t)FC * T_LEN * 4);
    memcpy(z, m_p, (size_t)FC * T_LEN * 4);

    // Flip channels
    {
        float *tmp = malloc((size_t)FC * T_LEN * 4);
        for (int c = 0; c < FC; c++)
            memcpy(&tmp[c * T_LEN], &z[(FC-1-c) * T_LEN], (size_t)T_LEN * 4);
        memcpy(z, tmp, (size_t)FC * T_LEN * 4);
        free(tmp);
    }

    // 4 blocks reverse order: idx 3, 2, 1, 0
    for (int fi = 3; fi >= 0; fi--) {
        int idx = fi;
        // pre: Conv1d(HP→FC, K=1) on first HP channels
        float *h = malloc((size_t)FC * T_LEN * 4);
        {
            // Extract first HP channels
            float *z_hp = malloc((size_t)HP * T_LEN * 4);
            for (int t = 0; t < T_LEN; t++)
                for (int c = 0; c < HP; c++)
                    z_hp[c * T_LEN + t] = z[c * T_LEN + t];
            // Conv1d K=1: out[co][t] = sum_ci W[co][ci] * in[ci][t]
            for (int t = 0; t < T_LEN; t++)
                for (int co = 0; co < FC; co++) {
                    float v = m->flow_pre_b[idx][co];
                    for (int ci = 0; ci < HP; ci++)
                        v += m->flow_pre_w[idx][co * HP + ci] * z_hp[ci * T_LEN + t];
                    h[co * T_LEN + t] = v;
                }
            free(z_hp);
        }

        // cond: Conv1d(GIN→2*FC*NL, K=1)
        int CC = 2 * FC * NL;
        float *cond = malloc(CC * 4);
        for (int co = 0; co < CC; co++) {
            float v = m->flow_cond_b[idx][co];
            for (int ci = 0; ci < GIN; ci++)
                v += m->flow_cond_w[idx][co * GIN + ci] * g_emb[ci];
            cond[co] = v;
        }

        // WaveNet
        float *output = calloc((size_t)FC * T_LEN, 4);
        for (int j = 0; j < NL; j++) {
            // in_layer: Conv1d(FC→2FC, K=5)
            int K = 5;
            float *x_in = malloc((size_t)2 * FC * T_LEN * 4);
            for (int t = 0; t < T_LEN; t++)
                for (int co = 0; co < 2*FC; co++) {
                    float v = m->flow_in_b[idx][j][co];
                    for (int k = 0; k < K; k++) {
                        int tt = t + k - (K/2);
                        if (tt < 0 || tt >= T_LEN) continue;
                        for (int ci = 0; ci < FC; ci++)
                            v += m->flow_in_w[idx][j][co * FC * K + ci * K + k] * h[ci * T_LEN + tt];
                    }
                    x_in[co * T_LEN + t] = v;
                }

            // fused_add_tanh_sigmoid_multiply
            for (int c = 0; c < FC; c++) {
                float cv1 = cond[j * 2 * FC + c];
                float cv2 = cond[j * 2 * FC + FC + c];
                for (int t = 0; t < T_LEN; t++) {
                    float a = x_in[c * T_LEN + t] + cv1;
                    float b = x_in[(FC+c) * T_LEN + t] + cv2;
                    h[c * T_LEN + t] = tanhf(a) / (1.0f + expf(-b));
                }
            }
            free(x_in);

            // res_skip: Conv1d(FC→rs_out, K=1)
            int rs_out = (j < NL-1) ? 2*FC : FC;
            float *res_skip = malloc((size_t)rs_out * T_LEN * 4);
            for (int t = 0; t < T_LEN; t++)
                for (int co = 0; co < rs_out; co++) {
                    float v = m->flow_rs_b[idx][j][co];
                    for (int ci = 0; ci < FC; ci++)
                        v += m->flow_rs_w[idx][j][co * FC + ci] * h[ci * T_LEN + t];
                    res_skip[co * T_LEN + t] = v;
                }

            if (j < NL - 1) {
                for (int c = 0; c < FC; c++)
                    for (int t = 0; t < T_LEN; t++) {
                        h[c * T_LEN + t] += res_skip[c * T_LEN + t];
                        output[c * T_LEN + t] += res_skip[(FC+c) * T_LEN + t];
                    }
            } else {
                for (int c = 0; c < FC; c++)
                    for (int t = 0; t < T_LEN; t++)
                        output[c * T_LEN + t] += res_skip[c * T_LEN + t];
            }
            free(res_skip);
        }
        memcpy(h, output, (size_t)FC * T_LEN * 4);
        free(output);

        // post: Conv1d(FC→HP, K=1)
        float *m_out = malloc((size_t)HP * T_LEN * 4);
        for (int t = 0; t < T_LEN; t++)
            for (int co = 0; co < HP; co++) {
                float v = m->flow_post_b[idx][co];
                for (int ci = 0; ci < FC; ci++)
                    v += m->flow_post_w[idx][co * FC + ci] * h[ci * T_LEN + t];
                m_out[co * T_LEN + t] = v;
            }
        free(h);

        // Reverse coupling: x1 -= m
        for (int c = 0; c < HP; c++)
            for (int t = 0; t < T_LEN; t++)
                z[(HP+c) * T_LEN + t] -= m_out[c * T_LEN + t];
        free(m_out);

        // Flip (if not last)
        if (fi > 0) {
            float *tmp = malloc((size_t)FC * T_LEN * 4);
            for (int c = 0; c < FC; c++)
                memcpy(&tmp[c * T_LEN], &z[(FC-1-c) * T_LEN], (size_t)T_LEN * 4);
            memcpy(z, tmp, (size_t)FC * T_LEN * 4);
            free(tmp);
        }
        free(cond);
    }

    // Save
    FILE *f = fopen("scratch/flow_ref_out.bin", "wb");
    fwrite(z, sizeof(float), FC * T_LEN, f);
    fclose(f);
    printf("saved ref: z[0..5]=%g %g %g %g %g %g\n", z[0],z[1],z[2],z[3],z[4],z[5]);

    free(z); free(m_p); free(logs_p); free(g_emb);
    rvc_free(m); free(m);
    return 0;
}

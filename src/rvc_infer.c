/**
 * @file    rvc_infer.c
 * @brief   RVC v2 forward pass — encoder, flow, decoder.
 */
#include "rvc.h"
#include "threads.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static double spcv_tms(void) {
#ifdef _WIN32
    LARGE_INTEGER f,t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
    return (double)t.QuadPart*1000.0/(double)f.QuadPart;
#else
    return (double)clock()*1000.0/CLOCKS_PER_SEC;
#endif
}

/* ================================================================== */
/* Helpers                                                            */
/* ================================================================== */

/* Conv1d: in[Cin,L] w[Cout,Cin,K] b[Cout] → out[Cout,L]  (parallel over out) */
typedef struct { const float *in, *w, *b; float *out; int Cin, L, Cout, K, dil; } conv_ctx;
static void conv_worker(int i0, int i1, void *arg) {
    conv_ctx *c = (conv_ctx *)arg;
    int pad = (c->K / 2) * c->dil;
    for (int i = i0; i < i1; i++) {
        int co = i / c->L, l = i - co * c->L;
        float s = c->b ? c->b[co] : 0;
        for (int ci = 0; ci < c->Cin; ci++) {
            const float *wp = &c->w[(size_t)co * c->Cin * c->K + ci * c->K];
            for (int k = 0; k < c->K; k++) {
                int p = l - pad + k * c->dil;
                if (p >= 0 && p < c->L) s += c->in[(size_t)ci * c->L + p] * wp[k];
            }
        }
        c->out[i] = s;
    }
}
static void conv1d(float *out, const float *in, const float *w, const float *b,
                   int Cin, int L, int Cout, int K) {
    conv_ctx c = { in, w, b, out, Cin, L, Cout, K, 1 };
    spcv_par_for(Cout * L, conv_worker, &c);
}

/* Conv1d with dilation: in[Cin,L] w[Cout,Cin,K] b[Cout] → out[Cout,L] */
static void conv1d_dil(float *out, const float *in, const float *w, const float *b,
                       int Cin, int L, int Cout, int K, int dil) {
    conv_ctx c = { in, w, b, out, Cin, L, Cout, K, dil };
    spcv_par_for(Cout * L, conv_worker, &c);
}

/*
 * weight_norm(dim=0) for Conv1d [Cout,Cin,K]:
 *   w[co,ci,k] = g[co] * v[co,ci,k] / ||v[co,:,:]||
 * Precompute per-co scale: scale[co] = g[co] / ||v[co,:,:]||
 */
static float *wn_scale_conv1d(const float *v, const float *g, int Cout, int Cin, int K) {
    float *sc = malloc((size_t)Cout * sizeof(float));
    for (int co = 0; co < Cout; co++) {
        double s = 0;
        const float *vp = &v[(size_t)co * Cin * K];
        for (int i = 0; i < Cin * K; i++) s += (double)vp[i] * vp[i];
        float norm = sqrtf((float)s);
        sc[co] = (g ? g[co] : 1.0f) / (norm > 0 ? norm : 1.0f);
    }
    return sc;
}

/* Conv1d with weight_norm: in[Cin,L] v[Cout,Cin,K] g[Cout] b[Cout] → out[Cout,L] */
typedef struct { const float *in, *v, *b; float *out, *sc; int Cin, L, Cout, K, dil; } convwn_ctx;
static void convwn_worker(int i0, int i1, void *arg) {
    convwn_ctx *c = (convwn_ctx *)arg;
    int pad = (c->K / 2) * c->dil;
    for (int i = i0; i < i1; i++) {
        int co = i / c->L, l = i - co * c->L;
        float s = 0;
        float scale = c->sc[co];
        for (int ci = 0; ci < c->Cin; ci++) {
            const float *wp = &c->v[(size_t)co * c->Cin * c->K + ci * c->K];
            for (int k = 0; k < c->K; k++) {
                int p = l - pad + k * c->dil;
                if (p >= 0 && p < c->L) s += c->in[(size_t)ci * c->L + p] * wp[k];
            }
        }
        c->out[i] = s * scale + (c->b ? c->b[co] : 0);
    }
}
static void conv1d_wn(float *out, const float *in, const float *v, const float *g,
                      const float *b, int Cin, int L, int Cout, int K) {
    float *sc = wn_scale_conv1d(v, g, Cout, Cin, K);
    convwn_ctx c = { in, v, b, out, sc, Cin, L, Cout, K, 1 };
    spcv_par_for(Cout * L, convwn_worker, &c);
    free(sc);
}

/* Conv1d with dilation + weight_norm */
static void conv1d_wn_dil(float *out, const float *in, const float *v, const float *g,
                          const float *b, int Cin, int L, int Cout, int K, int dil) {
    float *sc = wn_scale_conv1d(v, g, Cout, Cin, K);
    convwn_ctx c = { in, v, b, out, sc, Cin, L, Cout, K, dil };
    spcv_par_for(Cout * L, convwn_worker, &c);
    free(sc);
}

/*
 * ConvTranspose1d with weight_norm(dim=0):
 *   weight [Cin,Cout,K], g [Cin]
 *   w[ci,co,k] = g[ci] * v[ci,co,k] / ||v[ci,:,:]||
 *   out[co, t] += in[ci, l] * w[ci,co,k]  where t = l*stride + k - pad
 */
/* Gather form: out[co,t] = b[co] + sum_{ci,k} in[ci,(t-k+pad)/stride]*v[ci,co,k]*sc[ci]
 * (valid when (t-k+pad)>=0, (t-k+pad)%stride==0, (t-k+pad)/stride < L).
 * Identical math to the old scatter form, but each output (co,t) is independent,
 * so it parallelizes cleanly over the flattened output index. */
typedef struct { const float *in, *v, *b; float *out, *sc; int Cin, L, Cout, K, stride, pad, L_out; } ctn_ctx;
static void ctn_worker(int i0, int i1, void *arg) {
    ctn_ctx *c = (ctn_ctx *)arg;
    for (int i = i0; i < i1; i++) {
        int co = i / c->L_out, t = i - co * c->L_out;
        float s = c->b ? c->b[co] : 0;
        for (int ci = 0; ci < c->Cin; ci++) {
            float scale = c->sc[ci];
            const float *inp = &c->in[(size_t)ci * c->L];
            const float *vp  = &c->v[(size_t)ci * c->Cout * c->K];
            for (int k = 0; k < c->K; k++) {
                int num = t - k + c->pad;
                if (num < 0) continue;
                if (num % c->stride != 0) continue;
                int li = num / c->stride;
                if (li >= c->L) continue;
                s += inp[li] * vp[co * c->K + k] * scale;
            }
        }
        c->out[i] = s;
    }
}
static void conv_t1d_wn(float *out, const float *in, const float *v, const float *g,
                        const float *b, int Cin, int L, int Cout, int K, int stride) {
    int OL = L * stride;
    int pad = (K - stride) / 2;
    float *sc = malloc((size_t)Cin * sizeof(float));
    for (int ci = 0; ci < Cin; ci++) {
        double s = 0;
        const float *vp = &v[(size_t)ci * Cout * K];
        for (int i = 0; i < Cout * K; i++) s += (double)vp[i] * vp[i];
        sc[ci] = (g ? g[ci] : 1.0f) / (sqrtf((float)s) > 0 ? sqrtf((float)s) : 1.0f);
    }
    ctn_ctx c = { in, v, b, out, sc, Cin, L, Cout, K, stride, pad, OL };
    spcv_par_for(Cout * OL, ctn_worker, &c);
    free(sc);
}

/* LayerNorm over channel dim: x[C,L] → out[C,L] */
/* Python: transpose(1,-1) → F.layer_norm(channels) → transpose(1,-1) back */
static void layernorm(float *out, const float *x, const float *gamma, const float *beta,
                      int C, int L, float eps) {
    for (int l = 0; l < L; l++) {
        float mean = 0;
        for (int c = 0; c < C; c++) mean += x[(size_t)c * L + l];
        mean /= C;
        float var = 0;
        for (int c = 0; c < C; c++) { float d = x[(size_t)c * L + l] - mean; var += d * d; }
        var /= C;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < C; c++)
            out[(size_t)c * L + l] = (x[(size_t)c * L + l] - mean) * inv * gamma[c] + beta[c];
    }
}

static void leaky_relu_worker(int i0, int i1, void *x) {
    float *p = (float *)x;
    for (int i = i0; i < i1; i++) if (p[i] < 0) p[i] *= 0.1f;
}
static void leaky_relu(float *x, int n) {
    spcv_par_for(n, leaky_relu_worker, x);
}

/* noise_conv: src[OL] → out[nC,nL], stride conv  out[co,l]=b[co]+sum_k src[l*stride+k-pad]*w[co,k] */
typedef struct { const float *src, *w, *b; float *out; int nC, nL, nK, stride, pad, OL; } nc_ctx;
static void nc_worker(int i0, int i1, void *arg) {
    nc_ctx *c = (nc_ctx *)arg;
    for (int i = i0; i < i1; i++) {
        int co = i / c->nL, l = i - co * c->nL;
        float s = c->b ? c->b[co] : 0;
        for (int k = 0; k < c->nK; k++) {
            int in_idx = l * c->stride + k - c->pad;
            if (in_idx >= 0 && in_idx < c->OL)
                s += c->src[in_idx] * c->w[co * c->nK + k];
        }
        c->out[i] = s;
    }
}
static void noise_conv(float *out, const float *src, const float *w, const float *b,
                       int nC, int nL, int nK, int stride, int OL) {
    nc_ctx c = { src, w, b, out, nC, nL, nK, stride, stride / 2, OL };
    spcv_par_for(nC * nL, nc_worker, &c);
}

/* ================================================================== */
/* ResBlock1: 3× (leaky_relu → conv1(dil) → leaky_relu → conv2) + residual */
/* ================================================================== */
static void resblock1(float *x, int C, int L, int K,
                      const float *w1[3], const float *b1[3], const float *g1[3],
                      const float *w2[3], const float *b2[3], const float *g2[3],
                      const int *dilations) {
    float *h = malloc((size_t)C * L * sizeof(float));
    float *s = malloc((size_t)C * L * sizeof(float));
    float *xt = malloc((size_t)C * L * sizeof(float));
    for (int j = 0; j < 3; j++) {
        /* leaky_relu into xt (preserves x for residual) */
        for (int i = 0; i < C * L; i++) xt[i] = x[i] < 0 ? x[i] * 0.1f : x[i];
        /* conv1 with dilation + weight_norm */
        conv1d_wn_dil(h, xt, w1[j], g1[j], b1[j], C, L, C, K, dilations[j]);
        /* leaky_relu on h */
        leaky_relu(h, C * L);
        /* conv2 (dilation=1) + weight_norm */
        conv1d_wn(s, h, w2[j], g2[j], b2[j], C, L, C, K);
        /* residual */
        for (int i = 0; i < C * L; i++) x[i] += s[i];
    }
    free(h); free(s); free(xt);
}

/* ================================================================== */
/* Encoder: TextEncoder768 forward                                     */
/* ================================================================== */

/* MultiHeadAttention with relative position embeddings */
/* x: [C, L], c: [C, L] (context, same as x for self-attn) */
/* Returns: [C, L] */
static int g_mha_dump = 0;
static void mha(float *out, const float *x, const float *c,
                const float *qw, const float *qb,
                const float *kw, const float *kb,
                const float *vw, const float *vb,
                const float *ow, const float *ob,
                const float *relk, const float *relv,
                int C, int L, int H, int max_rel) {
    int hd = C / H;  /* head dim */
    int W = (max_rel - 1) / 2;  /* window half-size: emb_rel is [1, 2W+1, hd] */
    /* Q, K, V are Conv1d(C,C,1) = pointwise */
    float *Q = malloc((size_t)C * L * sizeof(float));
    float *K = malloc((size_t)C * L * sizeof(float));
    float *V = malloc((size_t)C * L * sizeof(float));
    /* Pointwise conv: out[co,l] = b[co] + sum_ci w[co,ci] * in[ci,l] */
    for (int co = 0; co < C; co++) {
        float bqv = qb[co], bkv = kb[co], bvv = vb[co];
        for (int l = 0; l < L; l++) {
            float sq = bqv, sk = bkv, sv = bvv;
            for (int ci = 0; ci < C; ci++) {
                sq += qw[co * C + ci] * x[ci * L + l];
                sk += kw[co * C + ci] * c[ci * L + l];
                sv += vw[co * C + ci] * c[ci * L + l];
            }
            Q[co * L + l] = sq; K[co * L + l] = sk; V[co * L + l] = sv;
        }
    }
    if (getenv("SPCV_DUMP") && g_mha_dump == 0) {
        char p[256];
        snprintf(p, sizeof(p), "scratch/mha_Q.bin"); FILE *f=fopen(p,"wb"); if(f){fwrite(Q,sizeof(float),(size_t)C*L,f);fclose(f);}
        snprintf(p, sizeof(p), "scratch/mha_K.bin"); f=fopen(p,"wb"); if(f){fwrite(K,sizeof(float),(size_t)C*L,f);fclose(f);}
        snprintf(p, sizeof(p), "scratch/mha_V.bin"); f=fopen(p,"wb"); if(f){fwrite(V,sizeof(float),(size_t)C*L,f);fclose(f);}
    }
    float scale = 1.0f / sqrtf((float)hd);
    float *ao = malloc((size_t)C * L * sizeof(float));
    memset(ao, 0, (size_t)C * L * sizeof(float));
    float *ldump_pre  = (getenv("SPCV_DUMP") && g_mha_dump == 0) ? calloc((size_t)L*L, sizeof(float)) : NULL;
    float *ldump_post = (getenv("SPCV_DUMP") && g_mha_dump == 0) ? calloc((size_t)L*L, sizeof(float)) : NULL;

    for (int h = 0; h < H; h++) {
        int hq = h * hd;
        for (int l1 = 0; l1 < L; l1++) {
            float mx = -1e9f;
            float *logits = malloc((size_t)L * sizeof(float));
            for (int l2 = 0; l2 < L; l2++) {
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += Q[(hq + d) * L + l1] * K[(hq + d) * L + l2];
                logits[l2] = s * scale;
                if (logits[l2] > mx) mx = logits[l2];
            }
            /* Relative key contribution: idx = W + (l2-l1), valid for |l1-l2| <= W */
            if (relk)
            for (int l2 = 0; l2 < L; l2++) {
                int idx = W + (l2 - l1);
                if (idx >= 0 && idx < max_rel)
                    for (int d = 0; d < hd; d++)
                        logits[l2] += Q[(hq + d) * L + l1] * relk[idx * hd + d] * scale;
            }
            if (ldump_pre && h == 0) memcpy(&ldump_pre[l1*L], logits, (size_t)L*sizeof(float));
            /* Softmax */
            float sum = 0;
            for (int l2 = 0; l2 < L; l2++) { logits[l2] = expf(logits[l2] - mx); sum += logits[l2]; }
            for (int l2 = 0; l2 < L; l2++) logits[l2] /= sum;
            if (ldump_post && h == 0) memcpy(&ldump_post[l1*L], logits, (size_t)L*sizeof(float));
            /* Weighted sum of V + relative V */
            for (int d = 0; d < hd; d++) {
                float acc = 0;
                for (int l2 = 0; l2 < L; l2++) {
                    acc += logits[l2] * V[(hq + d) * L + l2];
                    if (relv) {
                        int idx = W + (l2 - l1);
                        if (idx >= 0 && idx < max_rel)
                            acc += logits[l2] * relv[idx * hd + d];
                    }
                }
                ao[(hq + d) * L + l1] = acc;
            }
            free(logits);
        }
    }
    if (ldump_pre) {
        FILE *f=fopen("scratch/mha_ao.bin","wb"); if(f){fwrite(ao,sizeof(float),(size_t)C*L,f);fclose(f);}
        f=fopen("scratch/mha_ow.bin","wb"); if(f){fwrite(ow,sizeof(float),(size_t)C*C,f);fclose(f);}
        f=fopen("scratch/mha_ob.bin","wb"); if(f){fwrite(ob,sizeof(float),(size_t)C,f);fclose(f);}
    }
    /* Output conv: pointwise */
    for (int co = 0; co < C; co++) {
        float bv = ob[co];
        for (int l = 0; l < L; l++) {
            float s = bv;
            for (int ci = 0; ci < C; ci++)
                s += ow[co * C + ci] * ao[ci * L + l];
            out[co * L + l] = s;
        }
    }
    if (ldump_pre)  { FILE *f=fopen("scratch/mha_logits_pre.bin","wb"); if(f){fwrite(ldump_pre,sizeof(float),(size_t)L*L,f);fclose(f);} free(ldump_pre); }
    if (ldump_post) { FILE *f=fopen("scratch/mha_P.bin","wb"); if(f){fwrite(ldump_post,sizeof(float),(size_t)L*L,f);fclose(f);} free(ldump_post); }
    g_mha_dump++;
    free(Q); free(K); free(V); free(ao);
}

/* FFN: conv_1(C→FF, K) → GELU → conv_2(FF→C, K) */
static void ffn(float *out, const float *x,
               const float *w1, const float *b1,
               const float *w2, const float *b2,
               int C, int FF, int L, int K) {
    float *h = malloc((size_t)FF * L * sizeof(float));
    conv1d(h, x, w1, b1, C, L, FF, K);
    /* ReLU (activation=None in Python reference) */
    for (int i = 0; i < FF * L; i++)
        if (h[i] < 0) h[i] = 0;
    conv1d(out, h, w2, b2, FF, L, C, K);
    free(h);
}

/* Trace: dump a [C,L] float tensor to scratch/<name>.bin when SPCV_DUMP is set. */
static void dump_tensor(const char *name, const float *x, int C, int L) {
    if (!getenv("SPCV_DUMP")) return;
    char path[256];
    snprintf(path, sizeof(path), "scratch/%s.bin", name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(x, sizeof(float), (size_t)C * L, f); fclose(f); }
}

/* Full encoder forward: phone[T,ques] + pitch[T] → m_p[FC,L], logs_p[FC,L] */
static int encoder_forward(const rvc_model_t *m,
                           const float *phone, int T,
                           const int64_t *pitch,
                           float *m_p, float *logs_p) {
    int P = m->cfg.p_dim;       /* hidden channels (192) */
    int Q = m->cfg.ques;        /* embedding dim (768) */
    int FF = m->cfg.ff_dim;     /* FFN hidden (768) */
    int H = m->cfg.p_heads;     /* heads (2) */
    int NL = m->cfg.p_layers;   /* layers (6) */
    int FK = m->cfg.ffn_kernel; /* FFN kernel (3) */
    int FC = m->cfg.filter_channels; /* flow channels (192) */
    int max_rel = m->attn_max_rel;
    int hd = m->attn_head_dim;
    g_mha_dump = 0;

    /* 1. x = emb_phone(phone) + emb_pitch(pitch)  → [T, P] */
    float *x = malloc((size_t)P * T * sizeof(float));
    for (int t = 0; t < T; t++) {
        for (int p = 0; p < P; p++) {
            float v = m->emb_phone_b[p];
            for (int q = 0; q < Q; q++)
                v += m->emb_phone_w[p * Q + q] * phone[t * Q + q];
            /* pitch embedding */
            int pi = (int)pitch[t];
            if (pi < 0) pi = 0;
            if (pi >= 256) pi = 255;
            v += m->emb_pitch_w[pi * P + p];
            x[p * T + t] = v;
        }
    }
    /* 2. x = x * sqrt(P) */
    float sc = sqrtf((float)P);
    for (int i = 0; i < P * T; i++) x[i] *= sc;
    /* 3. x = leaky_relu(x) */
    leaky_relu(x, P * T);
    dump_tensor("enc_x_emb", x, P, T);

    /* 4. Encoder layers (Post-LN) */
    float *y = malloc((size_t)P * T * sizeof(float));
    for (int i = 0; i < NL; i++) {
        char nm[32]; snprintf(nm, sizeof(nm), "enc_x_L%d", i);
        /* Attention */
        mha(y, x, x,
            m->attn_qw[i], m->attn_qb[i],
            m->attn_kw[i], m->attn_kb[i],
            m->attn_vw[i], m->attn_vb[i],
            m->attn_ow[i], m->attn_ob[i],
            m->attn_relk[i], m->attn_relv[i],
            P, T, H, max_rel);
        if (i == 0) dump_tensor("enc_attn0", y, P, T);  /* mha output, layer 0 only */
        /* Post-LN: x = LN(x + attn) */
        for (int j = 0; j < P * T; j++) y[j] += x[j];
        if (i == 0) dump_tensor("enc_L0_preln1", y, P, T);
        layernorm(x, y, m->ln1_gamma[i], m->ln1_beta[i], P, T, 1e-5f);
        if (i == 0) dump_tensor("enc_L0_ln1", x, P, T);
        /* FFN */
        ffn(y, x, m->ffn1w[i], m->ffn1b[i], m->ffn2w[i], m->ffn2b[i], P, FF, T, FK);
        if (i == 0) dump_tensor("enc_L0_ffn", y, P, T);
        /* Post-LN: x = LN(x + ffn) */
        for (int j = 0; j < P * T; j++) y[j] += x[j];
        if (i == 0) dump_tensor("enc_L0_preln2", y, P, T);
        layernorm(x, y, m->ln2_gamma[i], m->ln2_beta[i], P, T, 1e-5f);
        dump_tensor(nm, x, P, T);
    }

    /* 5. Projection: Conv1d(P, 2*FC, 1) → [2*FC, T] */
    int P2 = 2 * FC;
    float *stats = malloc((size_t)P2 * T * sizeof(float));
    for (int co = 0; co < P2; co++) {
        float bv = m->proj_b[co];
        for (int t = 0; t < T; t++) {
            float s = bv;
            for (int ci = 0; ci < P; ci++)
                s += m->proj_w[co * P + ci] * x[ci * T + t];
            stats[co * T + t] = s;
        }
    }
    /* Split: m_p = stats[:FC], logs_p = stats[FC:] */
    memcpy(m_p, stats, (size_t)FC * T * sizeof(float));
    memcpy(logs_p, &stats[(size_t)FC * T], (size_t)FC * T * sizeof(float));

    free(x); free(y); free(stats);
    return 0;
}

/**
 * @brief  Public wrapper for the encoder forward (for testing / ggml comparison).
 *         Computes m_p[FC,T] and logs_p[FC,T] from phone[T,ques] + pitch[T].
 * @param  m        loaded RVC model (weights + config).
 * @param  phone    phone embeddings [T, ques] (row-major, time-major).
 * @param  T        sequence length.
 * @param  pitch    pitch indices [T] (int64, clamped to [0,255]).
 * @param  m_p      output mean [FC, T] (C-major).
 * @param  logs_p   output log-std [FC, T] (C-major).
 * @return 0 on success.
 */
int rvc_encoder_forward(const rvc_model_t *m,
                        const float *phone, int T,
                        const int64_t *pitch,
                        float *m_p, float *logs_p) {
    return encoder_forward(m, phone, T, pitch, m_p, logs_p);
}

/* ================================================================== */
/* Flow: 4× ResidualCouplingLayer (reverse order)                     */
/* ================================================================== */

/* WaveNet: n_layers sublayers with in_layers, res_skip_layers, cond */
static int g_wn_dump = 0;
static void wavenet(float *out, const float *x, const float *cond,
                    const float *in_w[3], const float *in_b[3], const float *in_g[3],
                    const float *rs_w[3], const float *rs_b[3], const float *rs_g[3],
                    int FC, int L, int n_layers) {
    int K = 5;  /* kernel size */
    float *h = malloc((size_t)FC * L * sizeof(float));
    memcpy(h, x, (size_t)FC * L * sizeof(float));
    float *output = calloc((size_t)FC * L, sizeof(float));

    for (int j = 0; j < n_layers; j++) {
        int Cout_in = 2 * FC;
        /* in_layer: Conv1d(FC→2FC, K=5) + weight_norm */
        float *x_in = malloc((size_t)Cout_in * L * sizeof(float));
        conv1d_wn(x_in, h, in_w[j], in_g[j], in_b[j], FC, L, Cout_in, K);
        if (g_wn_dump) { char fn[64]; sprintf(fn, "scratch/wn_xin_%d.bin", j); FILE *f = fopen(fn, "wb"); if (f) { fwrite(x_in, sizeof(float), (size_t)Cout_in * L, f); fclose(f); } }

        /* fused_add_tanh_sigmoid_multiply */
        float *acts = malloc((size_t)FC * L * sizeof(float));
        for (int c = 0; c < FC; c++) {
            float cv = cond ? cond[j * 2 * FC + c] : 0;
            float cv2 = cond ? cond[j * 2 * FC + FC + c] : 0;
            for (int l = 0; l < L; l++) {
                float a = x_in[c * L + l] + cv;
                float b = x_in[(FC + c) * L + l] + cv2;
                acts[c * L + l] = tanhf(a) / (1.0f + expf(-b));
            }
        }
        if (g_wn_dump) { char fn[64]; sprintf(fn, "scratch/wn_acts_%d.bin", j); FILE *f = fopen(fn, "wb"); if (f) { fwrite(acts, sizeof(float), (size_t)FC * L, f); fclose(f); } }
        free(x_in);

        /* res_skip: Conv1d(FC→rs_out, K=1) + weight_norm */
        int rs_out = (j < n_layers - 1) ? 2 * FC : FC;
        float *res_skip = malloc((size_t)rs_out * L * sizeof(float));
        conv1d_wn(res_skip, acts, rs_w[j], rs_g[j], rs_b[j], FC, L, rs_out, 1);
        if (g_wn_dump) { char fn[64]; sprintf(fn, "scratch/wn_rs_%d.bin", j); FILE *f = fopen(fn, "wb"); if (f) { fwrite(res_skip, sizeof(float), (size_t)rs_out * L, f); fclose(f); } }
        free(acts);

        if (j < n_layers - 1) {
            for (int c = 0; c < FC; c++)
                for (int l = 0; l < L; l++) {
                    h[c * L + l] += res_skip[c * L + l];
                    output[c * L + l] += res_skip[(FC + c) * L + l];
                }
        } else {
            for (int c = 0; c < FC; c++)
                for (int l = 0; l < L; l++)
                    output[c * L + l] += res_skip[c * L + l];
        }
        free(res_skip);
    }
    memcpy(out, output, (size_t)FC * L * sizeof(float));
    free(h); free(output);
}

/* Full flow forward (reverse) */
static int flow_forward(const rvc_model_t *m,
                        const float *m_p, const float *logs_p,
                        const float *g_emb,
                        float *z_out, int L, uint32_t *seed) {
    int FC = m->cfg.filter_channels;   /* 192 */
    int HP = FC / 2;                   /* 96 */
    int GIN = m->emb_g_dim;            /* 256 (speaker emb dim) */
    int NL = 3;                        /* n_layers in WN */

    /* VAE reparameterization: z = m_p + exp(logs_p) * rnd */
    float *z = malloc((size_t)FC * L * sizeof(float));
    uint32_t s = seed ? *seed : 12345;
    for (int c = 0; c < FC; c++) {
        for (int l = 0; l < L; l++) {
            /* Box-Muller for standard normal */
            s = s * 1103515245 + 12345;
            float u1 = (float)((s >> 8) & 0xFFFFF) / (float)(1 << 20);
            s = s * 1103515245 + 12345;
            float u2 = (float)((s >> 8) & 0xFFFFF) / (float)(1 << 20);
            if (u1 < 1e-10f) u1 = 1e-10f;
            float rnd = sqrtf(-2.0f * logf(u1)) * cosf((float)M_PI * 2.0f * u2);
            z[c * L + l] = m_p[c * L + l] + expf(logs_p[c * L + l]) * rnd * 0.66666f;
        }
    }
    if (seed) *seed = s;
    if (getenv("SPCV_DUMP")) { FILE *f = fopen("scratch/vae_z.bin", "wb"); if (f) { fwrite(z, sizeof(float), (size_t)FC * L, f); fclose(f); } }

    /* Real reverse flow order: Flip, RCL3, Flip, RCL2, Flip, RCL1, Flip, RCL0.
     * i.e. an initial channel flip BEFORE the first block, then a flip between
     * each pair of blocks. (modules.Flip reverses the channel dim.) */
    {
        float *tmp = malloc((size_t)FC * L * sizeof(float));
        for (int c = 0; c < FC; c++)
            memcpy(&tmp[c * L], &z[(FC - 1 - c) * L], (size_t)L * sizeof(float));
        memcpy(z, tmp, (size_t)FC * L * sizeof(float));
        free(tmp);
    }

    /* Process 4 flow blocks in reverse order (compact indices 0-3) */
    for (int fi = 3; fi >= 0; fi--) {
        int idx = fi;  /* compact index */
        /* pre: Conv1d(HP→FC, 1) */
        float *h = malloc((size_t)FC * L * sizeof(float));
        conv1d(h, z, m->flow_pre_w[idx], m->flow_pre_b[idx], HP, L, FC, 1);
        if (getenv("SPCV_DUMP") && fi == 3) { FILE *f = fopen("scratch/fb3_pre.bin", "wb"); if (f) { fwrite(h, sizeof(float), (size_t)FC * L, f); fclose(f); } }

        /* cond: Conv1d(GIN→2*FC*NL, 1) + weight_norm */
        int CC = 2 * FC * NL;
        float *cond = malloc((size_t)CC * sizeof(float));
        conv1d_wn(cond, g_emb, m->flow_cond_v[idx], m->flow_cond_g[idx],
                  m->flow_cond_b[idx], GIN, 1, CC, 1);

        /* WaveNet */
        g_wn_dump = (getenv("SPCV_DUMP") && fi == 3) ? 1 : 0;
        wavenet(h, h, cond,
                m->flow_in_w[idx], m->flow_in_b[idx], m->flow_in_g[idx],
                m->flow_rs_w[idx], m->flow_rs_b[idx], m->flow_rs_g[idx],
                FC, L, NL);
        if (getenv("SPCV_DUMP") && fi == 3) { FILE *f = fopen("scratch/fb3_wn.bin", "wb"); if (f) { fwrite(h, sizeof(float), (size_t)FC * L, f); fclose(f); } }
        free(cond);

        /* post: Conv1d(FC→HP, 1) — mean_only, so output is HP */
        float *m_out = malloc((size_t)HP * L * sizeof(float));
        conv1d(m_out, h, m->flow_post_w[idx], m->flow_post_b[idx], FC, L, HP, 1);
        if (getenv("SPCV_DUMP") && fi == 3) { FILE *f = fopen("scratch/fb3_post.bin", "wb"); if (f) { fwrite(m_out, sizeof(float), (size_t)HP * L, f); fclose(f); } }
        free(h);

        /* Reverse: x1 = x1 - m (logs=0 since mean_only) */
        for (int c = 0; c < HP; c++)
            for (int l = 0; l < L; l++)
                z[(HP + c) * L + l] -= m_out[c * L + l];
        free(m_out);

        /* Flip: reverse channel order (if not last block) */
        if (fi > 0) {
            float *tmp = malloc((size_t)FC * L * sizeof(float));
            for (int c = 0; c < FC; c++)
                memcpy(&tmp[c * L], &z[(FC - 1 - c) * L], (size_t)L * sizeof(float));
            memcpy(z, tmp, (size_t)FC * L * sizeof(float));
            free(tmp);
        }
        if (getenv("SPCV_DUMP")) { char fn[64]; sprintf(fn, "scratch/flow_blk_%d.bin", fi); FILE *f = fopen(fn, "wb"); if (f) { fwrite(z, sizeof(float), (size_t)FC * L, f); fclose(f); } }
    }
    memcpy(z_out, z, (size_t)FC * L * sizeof(float));
    free(z);
    return 0;
}

/* ================================================================== */
/* Decoder: GeneratorNSF forward                                       */
/* ================================================================== */

static int decoder_forward(const rvc_model_t *m,
                           const float *z, int L,
                           const float *pitchf,  /* f0 in Hz [L] */
                           float *out, int *out_len) {
    int FC = m->cfg.filter_channels;       /* 192 */
    int UP = m->cfg.upsample_initial;      /* 512 */
    int NU = m->cfg.n_upsample;            /* 4 */
    int NUP = 1;
    for (int i = 0; i < NU; i++) NUP *= m->cfg.upsample_rates[i];
    int sr = m->cfg.sr;                    /* 48000 */
    int OL = L * NUP;                      /* output length */
    int NK = 3;                            /* num_kernels (3 resblocks per stage) */

    if (getenv("SPCV_DEBUG")) fprintf(stderr, "[dec] start L=%d OL=%d NU=%d UP=%d\n", L, OL, NU, UP);

    /* --- m_source: SineGen + SourceModuleHnNSF (harmonic_num=0) --- */
    /* Correct algorithm (per porting guide):
     *   - nearest-interpolate f0 (zero-order hold, frame = t / NUP)
     *   - phase accumulates in cycles: phase += f0[frame] / sr
     *   - keep phase in [0,1) for numerical stability
     *   - clean = 0.1 * sin(2*pi*phase) * uv
     *   - noise = (uv*0.003 + (1-uv)*0.0333) * randn()
     *   - output = tanh(w * (clean + noise) + b)                       */
    float *har_source = malloc((size_t)OL * sizeof(float));
    float mw = m->m_source_w ? m->m_source_w[0] : 1.0f;
    float mb = m->m_source_b ? m->m_source_b[0] : 0.0f;
    int no_noise = (getenv("SPCV_NO_NOISE") != NULL);
    uint32_t rng = 54321;
    double phase = 0.0;
    for (int t = 0; t < OL; t++) {
        int frame = t / NUP;
        if (frame >= L) frame = L - 1;
        float f0 = pitchf ? pitchf[frame] : 0.0f;
        float uv = (f0 > 0.0f) ? 1.0f : 0.0f;
        if (uv > 0.0f) {
            phase += (double)f0 / (double)sr;
            phase -= floor(phase);
        }
        double clean = uv * (0.1 * sin(2.0 * M_PI * phase));
        double noise = 0.0;
        if (!no_noise) {
            double noise_amp = uv * 0.003 + (1.0 - uv) * (0.1 / 3.0);
            rng = rng * 1103515245u + 12345u;
            float u1 = (float)((rng >> 8) & 0xFFFFF) / (float)(1 << 20);
            rng = rng * 1103515245u + 12345u;
            float u2 = (float)((rng >> 8) & 0xFFFFF) / (float)(1 << 20);
            if (u1 < 1e-10f) u1 = 1e-10f;
            noise = noise_amp * (double)sqrtf(-2.0f * logf(u1)) * cosf((float)M_PI * 2.0f * u2);
        }
        har_source[t] = (float)tanh((double)mw * (clean + noise) + (double)mb);
    }
    /* --- conv_pre: Conv1d(FC→UP, K=7) --- */
    float *d = malloc((size_t)UP * L * sizeof(float));
    conv1d(d, z, m->dec_conv_pre_w, m->dec_conv_pre_b, FC, L, UP, 7);

    /* --- cond: x = x + cond(g)  (pointwise conv, broadcast over time) --- */
    if (m->dec_cond_w && m->emb_g) {
        int GIN = m->emb_g_dim;
        for (int co = 0; co < UP; co++) {
            float v = m->dec_cond_b[co];
            for (int ci = 0; ci < GIN; ci++)
                v += m->dec_cond_w[co * GIN + ci] * m->emb_g[ci];
            for (int l = 0; l < L; l++)
                d[co * L + l] += v;
        }
    }
    /* --- Upsampling stages --- */
    int cur_C = UP, cur_L = L;
    for (int u = 0; u < NU; u++) {
        int nC = UP / (1 << (u + 1));  /* output channels */
        int rate = m->cfg.upsample_rates[u];
        int nL = cur_L * rate;
        if (getenv("SPCV_DEBUG")) fprintf(stderr, "[dec] stage u=%d nC=%d rate=%d nL=%d (in C=%d L=%d)\n", u, nC, rate, nL, cur_C, cur_L);

        /* leaky_relu */
        leaky_relu(d, cur_C * cur_L);

        /* upsample: ConvTranspose1d + weight_norm */
        float *d2 = malloc((size_t)nC * nL * sizeof(float));
        conv_t1d_wn(d2, d, m->ups_w[u], m->ups_g[u], m->ups_b[u],
                    cur_C, cur_L, nC, m->cfg.upsample_kernels[u], rate);
        free(d);
        d = d2;
        cur_C = nC; cur_L = nL;
        if (getenv("SPCV_DEBUG")) { double rms=0; for(int i=0;i<cur_C*cur_L;i++) rms+=d[i]*d[i]; fprintf(stderr,"[dec] stage %d upsample RMS=%.6f C=%d L=%d\n", u, sqrt(rms/(cur_C*cur_L)), cur_C, cur_L); }

        /* noise_conv: Conv1d(1→nC, K, stride) on har_source */
        if (m->noise_w[u]) {
            int nK, stride;
            if (u + 1 < NU) {
                stride = 1;
                for (int i = u + 1; i < NU; i++) stride *= m->cfg.upsample_rates[i];
                nK = stride * 2;
            } else {
                nK = 1; stride = 1;
            }
            float *ns = malloc((size_t)nC * nL * sizeof(float));
            noise_conv(ns, har_source, m->noise_w[u], m->noise_b[u], nC, nL, nK, stride, OL);
            for (int i = 0; i < nC * nL; i++) d[i] += ns[i];
            free(ns);
            if (getenv("SPCV_DEBUG")) { double rms=0; for(int i=0;i<cur_C*cur_L;i++) rms+=d[i]*d[i]; fprintf(stderr,"[dec] stage %d noise_conv RMS=%.6f\n", u, sqrt(rms/(cur_C*cur_L))); }
        }

        /* ResBlocks (averaged) */
        float *sum = calloc((size_t)cur_C * cur_L, sizeof(float));
        for (int r = 0; r < NK; r++) {
            int rb_idx = u * NK + r;
            float *d3 = malloc((size_t)cur_C * cur_L * sizeof(float));
            memcpy(d3, d, (size_t)cur_C * cur_L * sizeof(float));
            int K = m->rb_kernels[rb_idx];
            int C = m->rb_channels[rb_idx];
            int dil[3] = {1, 3, 5};
            resblock1(d3, C, cur_L, K,
                      m->rb_conv1_w[rb_idx], m->rb_conv1_b[rb_idx], m->rb_conv1_g[rb_idx],
                      m->rb_conv2_w[rb_idx], m->rb_conv2_b[rb_idx], m->rb_conv2_g[rb_idx],
                      dil);
            for (int i = 0; i < cur_C * cur_L; i++) sum[i] += d3[i];
            free(d3);
        }
        for (int i = 0; i < cur_C * cur_L; i++) sum[i] /= NK;
        if (getenv("SPCV_DEBUG")) { double rms=0; for(int i=0;i<cur_C*cur_L;i++) rms+=sum[i]*sum[i]; fprintf(stderr,"[dec] stage %d resblocks RMS=%.6f\n", u, sqrt(rms/(cur_C*cur_L))); }
        free(d);
        d = sum;
        if (getenv("SPCV_DEBUG")) fprintf(stderr, "[dec] stage u=%d done (C=%d L=%d)\n", u, cur_C, cur_L);
    }

    /* --- Final: leaky_relu → conv_post → tanh --- */
    leaky_relu(d, cur_C * cur_L);
    /* conv_post: Conv1d(cur_C→1, K=7), no bias */
    conv1d(out, d, m->conv_post_w, NULL, cur_C, cur_L, 1, 7);
    /* tanh */
    for (int i = 0; i < cur_L; i++) out[i] = tanhf(out[i]);

    free(d); free(har_source);
    *out_len = cur_L;
    return 0;
}

/* ================================================================== */
/* Debug: per-segment RMS                                              */
/* ================================================================== */
static void dbg_rms(const char *name, const float *x, int C, int L) {
    if (!getenv("SPCV_DEBUG")) return;
    int seg = L / 10;
    if (seg < 1) seg = 1;
    fprintf(stderr, "[trace] %-18s C=%4d L=%6d | ", name, C, L);
    for (int s = 0; s < 10; s++) {
        int start = s * seg * C;
        int end = ((s + 1) * seg) * C;
        if (end > C * L) end = C * L;
        double sum = 0;
        int n = end - start;
        for (int i = start; i < end; i++) sum += (double)x[i] * x[i];
        fprintf(stderr, "%.3f ", sqrt(sum / n));
    }
    fprintf(stderr, "\n");
}

/* ================================================================== */
/* Main entry point                                                    */
/* ================================================================== */

int rvc_infer(const rvc_model_t *m,
              const float *phone, int T,
              const int64_t *pitch, const float *pitchf,
              const float *noise,
              float *out, int *out_len,
              uint32_t *seed) {
    (void)noise; (void)seed;
    int FC = m->cfg.filter_channels;
    float *z = malloc((size_t)FC * T * sizeof(float));
    /* timing */
    double _t_enc=0,_t_flow=0,_t_dec=0;
    const int _do_time = (getenv("SPCV_TIME") != NULL);
    double _tt = spcv_tms();

    /* Debug: read z from file (skip encoder+flow) */
    const char *zfile = getenv("SPCV_Z_FILE");
    if (zfile) {
        FILE *f = fopen(zfile, "rb");
        if (!f) { fprintf(stderr, "[err] cannot open %s\n", zfile); free(z); return -1; }
        if (fread(z, sizeof(float), (size_t)FC * T, f) != (size_t)FC * T) {
            fprintf(stderr, "[err] z file too small\n"); fclose(f); free(z); return -1;
        }
        fclose(f);
        fprintf(stderr, "[dbg] loaded z from %s\n", zfile);
    } else {
        /* 1. Encoder */
        float *m_p = malloc((size_t)FC * T * sizeof(float));
        float *logs_p = malloc((size_t)FC * T * sizeof(float));
        double _t1=_tt;
        if (encoder_forward(m, phone, T, pitch, m_p, logs_p) != 0) {
            free(m_p); free(logs_p); free(z);
            return -1;
        }
        _t_enc=spcv_tms()-_t1; if(_do_time) fprintf(stderr,"[time]   encoder        %8.1f ms\n",_t_enc);
        dbg_rms("m_p", m_p, FC, T);
        dbg_rms("logs_p", logs_p, FC, T);
        if (getenv("SPCV_DUMP")) {
            FILE *f = fopen("scratch/m_p.bin", "wb"); if (f) { fwrite(m_p, sizeof(float), (size_t)FC * T, f); fclose(f); }
            f = fopen("scratch/logs_p.bin", "wb"); if (f) { fwrite(logs_p, sizeof(float), (size_t)FC * T, f); fclose(f); }
        }
        /* 2. Flow (reverse) with speaker embedding */
        double _t2=spcv_tms();
        if (flow_forward(m, m_p, logs_p, m->emb_g, z, T, seed) != 0) {
            free(m_p); free(logs_p); free(z);
            return -1;
        }
        _t_flow=spcv_tms()-_t2; if(_do_time) fprintf(stderr,"[time]   flow           %8.1f ms\n",_t_flow);
        dbg_rms("flow_z", z, FC, T);
        if (getenv("SPCV_DUMP")) {
            FILE *f = fopen("scratch/flow_z.bin", "wb"); if (f) { fwrite(z, sizeof(float), (size_t)FC * T, f); fclose(f); }
        }
        free(m_p); free(logs_p);
    }

    /* 4. Decoder */
    double _t3=spcv_tms();
    if (decoder_forward(m, z, T, pitchf, out, out_len) != 0) {
        free(z);
        return -1;
    }
    _t_dec=spcv_tms()-_t3; if(_do_time) fprintf(stderr,"[time]   decoder        %8.1f ms\n",_t_dec);
    dbg_rms("output", out, 1, *out_len);
    free(z);
    return 0;
}

// ggml_rvc_enc.c — RVC v2 encoder reading weights from GGUF.
//
// Layout: [C, T] with C SLOW, T FAST: data[c*T + t]. (Matches pure-C reference.)
// Weights read from GGUF via cpu_tensor() helper.

#include "ggml_rvc_enc.h"
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Read a tensor's float data from a GGUF file (CPU). Caller frees.
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
// Q,K,V: [hd, T] layout data[c*T + t] (c slow, t fast).
// relk,relv: [hd, max_rel] layout data[d*max_rel + idx] (d slow, idx fast).
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
            for (int d = 0; d < hd; d++)
                s += Q[d * T + l1] * K[d * T + l2];
            logits[l2] = s * scale;
            if (logits[l2] > mx) mx = logits[l2];
        }
        if (relk)
            for (int l2 = 0; l2 < T; l2++) {
                int idx = W + (l2 - l1);
                if (idx >= 0 && idx < max_rel) {
                    float s = 0;
                    for (int d = 0; d < hd; d++)
                        s += Q[d * T + l1] * relk[idx * hd + d];
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
                    if (idx >= 0 && idx < max_rel)
                        acc += logits[l2] * relv[idx * hd + d];
                }
            }
            ao[d * T + l1] = acc;
        }
    }
}

int spcv_ggml_encoder(spcv_ggml_t *g,
                      const float *phone, int T,
                      const int64_t *pitch,
                      float *m_p, float *logs_p,
                      int P, int Q, int FF, int H, int NL, int FK, int FC,
                      int max_rel, int hd) {
    const char *gp = spcv_ggml_gguf_path(g);
    if (!gp) { fprintf(stderr, "enc: no gguf path\n"); return 1; }

    char nm[128];
    int layer, i, t, co, ci, l, k, h, d;

    // ---- Embedding ----
    // GGUF: emb_phone.weight [Q, P] ne[0]=Q(fast), ne[1]=P(slow) → data[q + p*Q]
    //       emb_phone.bias [P] → data[p]
    //       emb_pitch.weight [P, 256] ne[0]=P(fast), ne[1]=256(slow) → data[p + pi*P]
    float *phone_w = cpu_tensor(gp, "enc_p.emb_phone.weight");
    float *phone_b = cpu_tensor(gp, "enc_p.emb_phone.bias");
    float *pitch_w = cpu_tensor(gp, "enc_p.emb_pitch.weight");
    if (!phone_w || !pitch_w) { fprintf(stderr, "enc: missing emb weights\n"); return 1; }

    // x layout: [P, T] data[p*T + t] (p slow, t fast) — matches pure-C
    float *x = malloc((size_t)P * T * sizeof(float));
    for (t = 0; t < T; t++) {
        int pi = (int)pitch[t]; if (pi < 0) pi = 0; if (pi >= 256) pi = 255;
        for (co = 0; co < P; co++) {
            float v = phone_b ? phone_b[co] : 0.0f;
            for (ci = 0; ci < Q; ci++) v += phone_w[ci + co * Q] * phone[ci + t * Q];
            v += pitch_w[co + pi * P];
            x[co * T + t] = v;
        }
    }
    free(phone_w); free(phone_b); free(pitch_w);
    float sc = sqrtf((float)P);
    for (i = 0; i < P * T; i++) { x[i] *= sc; if (x[i] < 0) x[i] *= 0.1f; }

    // ---- Encoder layers ----
    float *y = malloc((size_t)P * T * sizeof(float));

    for (layer = 0; layer < NL; layer++) {
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.weight", layer);
        float *wq = cpu_tensor(gp, nm);  // [1, P, P] → data[ci + co*P] (K=1)
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.bias", layer);
        float *bq = cpu_tensor(gp, nm);  // [P]
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_k.weight", layer);
        float *wk = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_k.bias", layer);
        float *bk = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_v.weight", layer);
        float *wv = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_v.bias", layer);
        float *bv = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_o.weight", layer);
        float *wo = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_o.bias", layer);
        float *bo = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.emb_rel_k", layer);
        float *relk = cpu_tensor(gp, nm);  // [hd, max_rel] → data[d + idx*hd]
        snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.emb_rel_v", layer);
        float *relv = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_1.%d.gamma", layer);
        float *ln1w = cpu_tensor(gp, nm);  // [P]
        snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_1.%d.beta", layer);
        float *ln1b = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_1.weight", layer);
        float *ff1w = cpu_tensor(gp, nm);  // [FK, P, FF] → data[k + ci*FK + co*P*FK]
        snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_1.bias", layer);
        float *ff1b = cpu_tensor(gp, nm);  // [FF]
        snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_2.weight", layer);
        float *ff2w = cpu_tensor(gp, nm);  // [FK, FF, P] → data[k + ci*FK + co*FF*FK]
        snprintf(nm, sizeof(nm), "enc_p.encoder.ffn_layers.%d.conv_2.bias", layer);
        float *ff2b = cpu_tensor(gp, nm);  // [P]
        snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_2.%d.gamma", layer);
        float *ln2w = cpu_tensor(gp, nm);
        snprintf(nm, sizeof(nm), "enc_p.encoder.norm_layers_2.%d.beta", layer);
        float *ln2b = cpu_tensor(gp, nm);

        if (!wq || !wk || !wv || !wo) { fprintf(stderr, "enc: layer %d missing\n", layer); return 1; }

        // Q/K/V: x [P,T] → Q,K,V [P,T] (pointwise conv = linear)
        float *Qm = malloc((size_t)P * T * 4), *Km = malloc((size_t)P * T * 4), *Vm = malloc((size_t)P * T * 4);
        for (co = 0; co < P; co++)
            for (l = 0; l < T; l++) {
                float sq = bq[co], sk = bk[co], sv = bv[co];
                for (ci = 0; ci < P; ci++) {
                    float xv = x[ci * T + l];
                    sq += wq[ci + co * P] * xv;
                    sk += wk[ci + co * P] * xv;
                    sv += wv[ci + co * P] * xv;
                }
                Qm[co * T + l] = sq; Km[co * T + l] = sk; Vm[co * T + l] = sv;
            }

        // Attention (per-head)
        float *ao = malloc((size_t)P * T * 4);
        for (h = 0; h < H; h++) {
            int hq = h * hd;
            attn_head(ao + (size_t)hq * T, Qm + (size_t)hq * T, Km + (size_t)hq * T, Vm + (size_t)hq * T,
                      relk, relv,
                      hd, T, max_rel);
        }
        free(Qm); free(Km); free(Vm);

        // Output projection
        for (co = 0; co < P; co++)
            for (l = 0; l < T; l++) {
                float s = bo[co];
                for (ci = 0; ci < P; ci++) s += wo[ci + co * P] * ao[ci * T + l];
                y[co * T + l] = s;
            }
        free(ao);
        free(wq); free(bq); free(wk); free(bk); free(wv); free(bv);
        free(wo); free(bo); free(relk); free(relv);

        // Residual + LayerNorm 1
        for (i = 0; i < P * T; i++) y[i] += x[i];
        for (t = 0; t < T; t++) {
            float mean = 0; for (co = 0; co < P; co++) mean += y[co * T + t]; mean /= P;
            float var = 0; for (co = 0; co < P; co++) { float dd = y[co * T + t] - mean; var += dd * dd; } var /= P;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            for (co = 0; co < P; co++) x[co * T + t] = (y[co * T + t] - mean) * inv * ln1w[co] + ln1b[co];
        }
        free(ln1w); free(ln1b);

        // FFN: conv1(K=FK) → ReLU → conv2(K=FK)
        // GGUF conv weight: [K, IC, OC] → data[k + ci*K + co*IC*K]
        int pad = FK / 2;
        float *h1 = malloc((size_t)FF * T * 4);
        for (co = 0; co < FF; co++)
            for (l = 0; l < T; l++) {
                float s = ff1b[co];
                for (k = 0; k < FK; k++) {
                    int li = l + k - pad;
                    if (li < 0 || li >= T) continue;
                    for (ci = 0; ci < P; ci++)
                        s += ff1w[k + ci * FK + co * P * FK] * x[ci * T + li];
                }
                h1[co * T + l] = s;
            }
        for (i = 0; i < FF * T; i++) if (h1[i] < 0) h1[i] = 0;
        for (co = 0; co < P; co++)
            for (l = 0; l < T; l++) {
                float s = ff2b[co];
                for (k = 0; k < FK; k++) {
                    int li = l + k - pad;
                    if (li < 0 || li >= T) continue;
                    for (ci = 0; ci < FF; ci++)
                        s += ff2w[k + ci * FK + co * FF * FK] * h1[ci * T + li];
                }
                y[co * T + l] = s;
            }
        free(h1); free(ff1w); free(ff1b); free(ff2w); free(ff2b);

        // Residual + LayerNorm 2
        for (i = 0; i < P * T; i++) y[i] += x[i];
        for (t = 0; t < T; t++) {
            float mean = 0; for (co = 0; co < P; co++) mean += y[co * T + t]; mean /= P;
            float var = 0; for (co = 0; co < P; co++) { float dd = y[co * T + t] - mean; var += dd * dd; } var /= P;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            for (co = 0; co < P; co++) x[co * T + t] = (y[co * T + t] - mean) * inv * ln2w[co] + ln2b[co];
        }
        free(ln2w); free(ln2b);
    }

    // ---- Projection: [P, T] → [2*FC, T] ----
    float *pw = cpu_tensor(gp, "enc_p.proj.weight");  // [1, P, 2FC] → data[ci + co*P]
    float *pb = cpu_tensor(gp, "enc_p.proj.bias");    // [2FC]
    if (!pw || !pb) { fprintf(stderr, "enc: missing proj\n"); return 1; }
    int P2 = 2 * FC;
    float *stats = malloc((size_t)P2 * T * 4);
    for (co = 0; co < P2; co++)
        for (l = 0; l < T; l++) {
            float s = pb[co];
            for (ci = 0; ci < P; ci++) s += pw[ci + co * P] * x[ci * T + l];
            stats[co * T + l] = s;
        }
    free(pw); free(pb);
    // Output layout: [FC, T] data[c*T + t] — same as pure-C
    memcpy(m_p, stats, (size_t)FC * T * 4);
    memcpy(logs_p, stats + FC * T, (size_t)FC * T * 4);
    free(stats); free(x); free(y);
    return 0;
}

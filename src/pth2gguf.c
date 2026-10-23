// pth2gguf.c — in-app, in-memory .pth → GGUF conversion (parallelized).
// See pth2gguf.h for the contract. Spec: tools/pth-to-gguf.md.
//
// Pipeline:
//   1. normalize names, classify weight-norm pairs
//   2. serial: create ggml tensors (fast pointer-bump allocation)
//   3. PARALLEL: fill tensor data (I/O from .pth + dtype convert + weight-norm fusion)
//   4. serial: register in gguf table
//
// All math in float32. weight-norm fused in PyTorch layout, then reversed.

#include "pth2gguf.h"
#include "threads.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_NAME 256

// ---------------------------------------------------------------------------
// name normalization
// ---------------------------------------------------------------------------

static void strip_prefixes(char *name) {
    static const char *prefixes[] = {
        "module.", "model.", "net_g.", "synthesizer.", "generator."
    };
    for (;;) {
        int changed = 0;
        for (int i = 0; i < 5; i++) {
            size_t n = strlen(prefixes[i]);
            if (strncmp(name, prefixes[i], n) == 0) {
                memmove(name, name + n, strlen(name + n) + 1);
                changed = 1;
            }
        }
        if (!changed) break;
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static int64_t t_n_elem(const pth_tensor_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

// ---------------------------------------------------------------------------
// weight-norm fusion (spec 5.2)
// ---------------------------------------------------------------------------

static void fuse_weight_norm_into(float *W, const pth_tensor_t *tv,
                                  const float *V, const float *G) {
    int64_t d0 = tv->shape[0];
    int64_t rest = t_n_elem(tv) / d0;
    for (int64_t i = 0; i < d0; i++) {
        double ss = 0.0;
        for (int64_t j = 0; j < rest; j++) { double v = V[i*rest+j]; ss += v*v; }
        float norm = (float)sqrt(ss);
        float scale = (norm == 0.0f) ? 0.0f : G[i] / norm;
        for (int64_t j = 0; j < rest; j++) W[i*rest+j] = V[i*rest+j] * scale;
    }
}

// ---------------------------------------------------------------------------
// parallel worker context
// ---------------------------------------------------------------------------

typedef struct {
    const pth_model_t *m;
    // output: ggml tensor data pointers (pre-allocated)
    float *dst[PK_MAX_TENSORS];
    // tensor indices to process (regular + fused)
    int    idx[PK_MAX_TENSORS];
    int    n_items;
    // for fused: the _g tensor index (-1 for regular)
    int    g_idx[PK_MAX_TENSORS];
    // kind: 0=regular, 2=fused
    int    kind[PK_MAX_TENSORS];
} par_ctx_t;

static void fill_worker(int begin, int end, void *ctx_) {
    par_ctx_t *ctx = (par_ctx_t *)ctx_;
    const pth_model_t *m = ctx->m;
    for (int i = begin; i < end; i++) {
        int ti = ctx->idx[i];
        float *dst = ctx->dst[i];
        const pth_tensor_t *t = &m->pkl.tensors[ti];
        int64_t n = t_n_elem(t);

        if (ctx->kind[i] == 0) {
            // regular: just read + convert
            if (pth_read_tensor_f32(m, t, dst) != 0) {
                fprintf(stderr, "[pth2gguf] thread: read failed for tensor %d\n", ti);
                return;
            }
        } else {
            // fused: read V and G, then fuse into dst
            int gi = ctx->g_idx[i];
            const pth_tensor_t *tg = &m->pkl.tensors[gi];
            float *V = (float *)malloc((size_t)n * sizeof(float));
            float *G = (float *)malloc((size_t)t_n_elem(tg) * sizeof(float));
            if (!V || !G) { free(V); free(G); return; }
            if (pth_read_tensor_f32(m, t, V) != 0) { free(V); free(G); return; }
            if (pth_read_tensor_f32(m, tg, G) != 0) { free(V); free(G); return; }
            fuse_weight_norm_into(dst, t, V, G);
            free(V); free(G);
        }
    }
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------

static void set_metadata(struct gguf_context * gg, const pth_model_t *m) {
    gguf_set_val_str(gg, "general.architecture", "rvc");
    gguf_set_val_str(gg, "general.name",        "rvc-model");
    gguf_set_val_u32(gg, "general.alignment",   32);
    gguf_set_val_str(gg, "rvc.version",         "v1");
    gguf_set_val_str(gg, "rvc.weight_norm",     "fused");
    gguf_set_val_u32(gg, "rvc.causal",          0);

    if (!m->pkl.has_config) return;
    const pth_config_t *c = &m->pkl.config;
    gguf_set_val_u32(gg, "rvc.sample_rate",      (uint32_t)c->sr);
    gguf_set_val_u32(gg, "rvc.hidden_channels",  (uint32_t)c->p_dim);
    gguf_set_val_u32(gg, "rvc.filter_channels",  (uint32_t)c->filter_channels);
    gguf_set_val_u32(gg, "rvc.n_heads",          (uint32_t)c->p_heads);
    gguf_set_val_u32(gg, "rvc.n_layers",         (uint32_t)c->p_layers);
    gguf_set_val_u32(gg, "rvc.kernel_size",      (uint32_t)c->p_kernel);
    gguf_set_val_u32(gg, "rvc.upsample_initial", (uint32_t)c->upsample_initial);
    gguf_set_val_u32(gg, "rvc.frame_rate",       (uint32_t)c->frame_rate);
    if (c->version[0]) gguf_set_val_str(gg, "rvc.model_version", c->version);
}

// ---------------------------------------------------------------------------
// main conversion
// ---------------------------------------------------------------------------

int spcv_pth2gguf(const pth_model_t *m, spcv_gguf_t *out) {
    out->gctx = NULL; out->ggctx = NULL;

    pickle_ctx_t *pkl = &m->pkl;
    int n = pkl->n_tensors;
    if (n <= 0) { fprintf(stderr, "[pth2gguf] no tensors\n"); return -1; }

    // Size the ggml memory pool.
    size_t total = 0;
    for (int i = 0; i < n; i++)
        total += (size_t)t_n_elem(&pkl->tensors[i]) * sizeof(float);
    size_t mem = total + (size_t)n * 512 + (8u << 20);

    struct ggml_init_params ip = { mem, NULL, /*no_alloc=*/false };
    struct ggml_context * gctx = ggml_init(ip);
    if (!gctx) { fprintf(stderr, "[pth2gguf] ggml_init failed\n"); return -1; }

    struct gguf_context * gg = gguf_init_empty();
    if (!gg) { fprintf(stderr, "[pth2gguf] gguf_init_empty failed\n"); ggml_free(gctx); return -1; }

    set_metadata(gg, m);

    // Pass 1: normalize names + classify.
    char names[PK_MAX_TENSORS][MAX_NAME];
    int  kind[PK_MAX_TENSORS];
    char base[PK_MAX_TENSORS][MAX_NAME];
    for (int i = 0; i < n; i++) {
        const pth_tensor_t *t = &pkl->tensors[i];
        snprintf(names[i], MAX_NAME, "%s", t->name);
        char *nm = names[i];
        strip_prefixes(nm);
        size_t len = strlen(nm);
        if (len > 9 && strcmp(nm + len - 9, ".weight_g") == 0) {
            kind[i] = 1; nm[len - 9] = '\0';
            snprintf(base[i], MAX_NAME, "%s", nm);
        } else if (len > 9 && strcmp(nm + len - 9, ".weight_v") == 0) {
            kind[i] = 2; nm[len - 9] = '\0';
            snprintf(base[i], MAX_NAME, "%s", nm);
        } else {
            kind[i] = 0;
        }
    }

    // Build work list: regular tensors + fused pairs (skip _g and _v originals).
    par_ctx_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.m = m;
    pc.n_items = 0;

    // Regular tensors
    for (int i = 0; i < n; i++) {
        if (kind[i] != 0) continue;
        pc.idx[pc.n_items] = i;
        pc.kind[pc.n_items] = 0;
        pc.g_idx[pc.n_items] = -1;
        pc.n_items++;
    }
    // Fused pairs (iterate _v side)
    for (int i = 0; i < n; i++) {
        if (kind[i] != 2) continue;
        int g = -1;
        for (int j = 0; j < n; j++)
            if (kind[j] == 1 && strcmp(base[j], base[i]) == 0) { g = j; break; }
        if (g < 0) continue;
        pc.idx[pc.n_items] = i;
        pc.kind[pc.n_items] = 2;
        pc.g_idx[pc.n_items] = g;
        pc.n_items++;
    }

    // Pass 2 (serial): create ggml tensors, collect output pointers.
    struct ggml_tensor *gt_list[PK_MAX_TENSORS];
    int n_regular = 0, n_fused = 0;
    for (int i = 0; i < pc.n_items; i++) {
        int ti = pc.idx[i];
        const pth_tensor_t *t = &pkl->tensors[ti];
        int64_t ne[4] = { 1, 1, 1, 1 };
        int nd = t->ndim > 4 ? 4 : t->ndim;
        for (int d = 0; d < nd; d++) ne[d] = t->shape[t->ndim - 1 - d];

        char out_name[MAX_NAME];
        if (pc.kind[i] == 2)
            snprintf(out_name, MAX_NAME, "%s.weight", base[ti]);
        else
            snprintf(out_name, MAX_NAME, "%s", names[ti]);

        struct ggml_tensor *gt = ggml_new_tensor(gctx, GGML_TYPE_F32, nd, ne);
        if (!gt) goto fail;
        ggml_set_name(gt, out_name);
        gt_list[i] = gt;
        pc.dst[i] = (float *)gt->data;

        if (pc.kind[i] == 2) n_fused++; else n_regular++;
    }

    // Pass 3 (PARALLEL): fill tensor data.
    spcv_par_for(pc.n_items, fill_worker, &pc);

    // Pass 4 (serial): register in gguf table.
    for (int i = 0; i < pc.n_items; i++)
        gguf_add_tensor(gg, gt_list[i]);

    fprintf(stderr, "[pth2gguf] converted %d tensors (%d regular, %d fused) [%d threads]\n",
            n_regular + n_fused, n_regular, n_fused, spcv_nthreads());
    out->gctx = gctx;
    out->ggctx = gg;
    return 0;

fail:
    fprintf(stderr, "[pth2gguf] conversion FAILED\n");
    gguf_free(gg);
    ggml_free(gctx);
    out->gctx = NULL; out->ggctx = NULL;
    return -1;
}

void spcv_gguf_free(spcv_gguf_t *m) {
    if (!m) return;
    if (m->ggctx) gguf_free(m->ggctx);
    if (m->gctx)  ggml_free(m->gctx);
    m->gctx = NULL; m->ggctx = NULL;
}

struct ggml_tensor * spcv_gguf_tensor(const spcv_gguf_t *m, const char *name) {
    if (!m || !m->gctx) return NULL;
    return ggml_get_tensor(m->gctx, name);
}

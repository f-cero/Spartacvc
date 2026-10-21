// pth2gguf.c — in-app, in-memory .pth → GGUF conversion.
// See pth2gguf.h for the contract. Spec: tools/pth-to-gguf.md.
//
// Pipeline per tensor:
//   1. normalize name (strip wrapper prefixes)
//   2. if weight-norm pair: fuse  W[i,...] = V[i,...] * G[i] / ||V[i,...]||
//   3. reverse dims (PyTorch [d0..dn] -> ggml ne [dn..d0]); flat byte copy
//   4. add to ggml context + gguf tensor table
//
// All math in float32. weight-norm fused in PyTorch layout, then reversed.

#include "pth2gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_NAME 256

// ---------------------------------------------------------------------------
// name normalization
// ---------------------------------------------------------------------------

// Strip leading wrapper prefixes (module., model., net_g., synthesizer.,
// generator.) repeatedly. Operates in place. (spec 1.2)
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
// small helpers
// ---------------------------------------------------------------------------

static int64_t t_n_elem(const pth_tensor_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

// Create a ggml tensor in gctx with reversed dims, copy flat f32 data,
// register it in the gguf tensor table. (spec 2: reverse dims, copy flat)
static struct ggml_tensor * add_tensor(
        struct ggml_context * gctx,
        struct gguf_context * ggctx,
        const char *name,
        const pth_tensor_t *t,
        const float *data)
{
    int64_t ne[4] = { 1, 1, 1, 1 };
    int nd = t->ndim > 4 ? 4 : t->ndim;
    for (int i = 0; i < nd; i++) ne[i] = t->shape[t->ndim - 1 - i];

    struct ggml_tensor * gt = ggml_new_tensor(gctx, GGML_TYPE_F32, nd, ne);
    if (!gt) return NULL;
    ggml_set_name(gt, name);
    memcpy(gt->data, data, (size_t)t_n_elem(t) * sizeof(float));
    gguf_add_tensor(ggctx, gt);
    return gt;
}

// ---------------------------------------------------------------------------
// weight-norm fusion (spec 5.2)
// ---------------------------------------------------------------------------

// Fuse V (weight_v) and G (weight_g) into W = V * (G / ||V||) along dim 0.
// Both inputs in PyTorch layout (before reversal). Returns malloc'd f32.
static float * fuse_weight_norm(const pth_tensor_t *tv, const float *V,
                                const float *G) {
    int64_t d0 = tv->shape[0];
    int64_t rest = t_n_elem(tv) / d0;
    float *W = (float *)malloc((size_t)t_n_elem(tv) * sizeof(float));
    if (!W) return NULL;
    for (int64_t i = 0; i < d0; i++) {
        double ss = 0.0;
        for (int64_t j = 0; j < rest; j++) { double v = V[i*rest+j]; ss += v*v; }
        float norm = (float)sqrt(ss);
        float scale = (norm == 0.0f) ? 0.0f : G[i] / norm;
        for (int64_t j = 0; j < rest; j++) W[i*rest+j] = V[i*rest+j] * scale;
    }
    return W;
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

    // Size the ggml memory pool: all tensor data + per-tensor struct overhead.
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

    // Pass 1: normalize names + classify weight-norm parts.
    // (fixed-size arrays: MSVC has no VLAs; PK_MAX_TENSORS bounds n)
    char names[PK_MAX_TENSORS][MAX_NAME];
    int  kind[PK_MAX_TENSORS];        // 0=regular, 1=weight_g, 2=weight_v
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

    // Pass 2: emit regular tensors.
    int n_regular = 0, n_fused = 0;
    for (int i = 0; i < n; i++) {
        if (kind[i] != 0) continue;
        const pth_tensor_t *t = &pkl->tensors[i];
        float *buf = (float *)malloc((size_t)t_n_elem(t) * sizeof(float));
        if (!buf) goto fail;
        if (pth_read_tensor_f32(m, t, buf) != 0) { free(buf); goto fail; }
        if (!add_tensor(gctx, gg, names[i], t, buf)) { free(buf); goto fail; }
        free(buf);
        n_regular++;
    }

    // Pass 3: emit fused weight-norm tensors (base.weight).
    for (int i = 0; i < n; i++) {
        if (kind[i] != 2) continue;                 // iterate the _v side
        // find matching _g for the same base
        int g = -1;
        for (int j = 0; j < n; j++)
            if (kind[j] == 1 && strcmp(base[j], base[i]) == 0) { g = j; break; }
        if (g < 0) { fprintf(stderr, "[pth2gguf] weight_v without weight_g: %s\n", base[i]); continue; }

        const pth_tensor_t *tv = &pkl->tensors[i];
        const pth_tensor_t *tg = &pkl->tensors[g];
        float *V = (float *)malloc((size_t)t_n_elem(tv) * sizeof(float));
        float *G = (float *)malloc((size_t)t_n_elem(tg) * sizeof(float));
        if (!V || !G) { free(V); free(G); goto fail; }
        if (pth_read_tensor_f32(m, tv, V) != 0) { free(V); free(G); goto fail; }
        if (pth_read_tensor_f32(m, tg, G) != 0) { free(V); free(G); goto fail; }

        float *W = fuse_weight_norm(tv, V, G);
        free(V); free(G);
        if (!W) goto fail;

        char out_name[MAX_NAME];
        snprintf(out_name, MAX_NAME, "%s.weight", base[i]);
        if (!add_tensor(gctx, gg, out_name, tv, W)) { free(W); goto fail; }
        free(W);
        n_fused++;
    }

    fprintf(stderr, "[pth2gguf] converted %d tensors (%d regular, %d fused)\n",
            n_regular + n_fused, n_regular, n_fused);
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

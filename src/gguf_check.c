// gguf_check.c — verification of converted .gguf RVC weight files.
// See gguf_check.h for the contract.
//
// Loads the .gguf (tensor data allocated in a ggml context), checks
// structure/metadata, and — when given the source .pth — verifies that every
// gguf tensor's shape equals the reversed pth shape, that no weight-norm parts
// leaked through, and spot-checks fused values + flat-copy data integrity.

#include "gguf_check.h"
#include "pth_loader.h"
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fails = 0, g_checks = 0;

/** Report one check result. */
static void CHECK(int cond, const char *msg) {
    g_checks++;
    if (cond) printf("  PASS: %s\n", msg);
    else { printf("  FAIL: %s\n", msg); g_fails++; }
}

/** Find a gguf tensor id by name, or -1. */
static int64_t find_tensor(const struct gguf_context *gg, const char *name) {
    for (int64_t i = 0; i < gguf_get_n_tensors(gg); i++)
        if (strcmp(gguf_get_tensor_name(gg, i), name) == 0) return i;
    return -1;
}

/**
 * Verify one gguf tensor's shape == reversed pth shape.
 * @param fused  if true, the pth tensor is "<name>_v" (weight-norm value part)
 */
static void check_shape(const struct gguf_context *gg, pth_model_t *m,
                        const char *name, int fused) {
    int64_t id = find_tensor(gg, name);
    if (id < 0) { CHECK(0, name); return; }
    const int64_t *gne = gguf_get_tensor_ne(gg, id);

    const pth_tensor_t *pt = pth_find_tensor(m, name);
    if (!pt && fused) {
        char nv[512]; snprintf(nv, sizeof(nv), "%s_v", name);
        pt = pth_find_tensor(m, nv);
    }
    if (!pt) {
        char msg[600]; snprintf(msg, sizeof(msg), "%s: found in pth", name);
        CHECK(0, msg);
        return;
    }
    int nd = pt->ndim;
    int ok = (nd <= 4);
    for (int i = 0; ok && i < nd; i++)
        if (gne[i] != pt->shape[nd - 1 - i]) ok = 0;
    char msg[600];
    snprintf(msg, sizeof(msg), "%s: gguf ne=[%lld,%lld,%lld] == reversed pth [%s]",
             name, (long long)gne[0], (long long)gne[1], (long long)gne[2],
             (fused ? "(weight_v) " : "") );
    (void)nd; (void)pt;
    CHECK(ok, msg);
}

/** Verify a fused weight-norm value: W[idx] == V[idx]*G[idx]/||V row||. */
static void check_fused_value(const struct ggml_context *gctx, pth_model_t *m,
                              const char *base) {
    const pth_tensor_t *tv = pth_find_tensor(m, base); // base = "...weight_v"
    // derive the matching weight_g name: strip trailing "_v", append "_g"
    char gv[512];
    snprintf(gv, sizeof(gv), "%.*s_g", (int)(strlen(base) - 2), base);
    const pth_tensor_t *tg = pth_find_tensor(m, gv);
    if (!tv || !tg) { char mm[600]; snprintf(mm,sizeof(mm),"%s / %s in pth", base, gv); CHECK(0, mm); return; }

    int64_t n_v = 1; for (int i = 0; i < tv->ndim; i++) n_v *= tv->shape[i];
    int64_t rest = n_v / tv->shape[0];
    float *V = malloc((size_t)n_v * 4), *G = malloc((size_t)n_v * 4);
    pth_read_tensor_f32(m, tv, V);
    pth_read_tensor_f32(m, tg, G);
    double ss = 0; for (int64_t j = 0; j < rest; j++) ss += (double)V[j] * V[j];
    float expect = V[0] * (G[0] / (float)sqrt(ss));
    // gguf name = base with trailing "_v" stripped
    char gn[512]; snprintf(gn, sizeof(gn), "%.*s", (int)(strlen(base) - 2), base);
    struct ggml_tensor *gt = ggml_get_tensor(gctx, gn);
    float got = gt ? ((const float *)gt->data)[0] : NAN;
    char msg[600];
    snprintf(msg, sizeof(msg), "%s: fused[0] gguf=%.6f expect=%.6f", gn, got, expect);
    CHECK(gt && fabsf(got - expect) < 1e-5f, msg);
    free(V); free(G);
}

/** Verify a regular tensor's data is byte-identical to the pth. */
static void check_data_identity(const struct ggml_context *gctx, pth_model_t *m,
                                const char *name) {
    const pth_tensor_t *pt = pth_find_tensor(m, name);
    if (!pt) { CHECK(0, name); return; }
    int64_t n = 1; for (int i = 0; i < pt->ndim; i++) n *= pt->shape[i];
    float *P = malloc((size_t)n * 4);
    pth_read_tensor_f32(m, pt, P);
    struct ggml_tensor *gt = ggml_get_tensor(gctx, name);
    int ok = gt != NULL;
    if (ok) {
        const float *G = (const float *)gt->data;
        for (int64_t i = 0; i < n; i++) if (P[i] != G[i]) { ok = 0; break; }
    }
    char msg[600]; snprintf(msg, sizeof(msg), "%s: data byte-identical (%lld elems)", name, (long long)n);
    CHECK(ok, msg);
    free(P);
}

int spcv_gguf_check(const char *gguf_path, const char *pth_path) {
    g_fails = 0; g_checks = 0;
    printf("== gguf check: %s ==\n", gguf_path);

    /* ---- load ---- */
    struct ggml_context *gctx = NULL;
    struct gguf_init_params params = { .no_alloc = false, .ctx = &gctx };
    struct gguf_context *gg = gguf_init_from_file(gguf_path, params);
    if (!gg) { fprintf(stderr, "  FAIL: cannot open gguf file\n"); return 1; }
    CHECK(1, "gguf file opens");

    int64_t n = gguf_get_n_tensors(gg);
    printf("  tensors: %lld\n", (long long)n);
    CHECK(n > 0, "has tensors");

    /* ---- metadata ---- */
    int64_t k = gguf_find_key(gg, "general.architecture");
    CHECK(k >= 0 && strcmp(gguf_get_val_str(gg, k), "rvc") == 0,
          "general.architecture == \"rvc\"");
    k = gguf_find_key(gg, "rvc.weight_norm");
    CHECK(k >= 0 && strcmp(gguf_get_val_str(gg, k), "fused") == 0,
          "rvc.weight_norm == \"fused\"");

    /* ---- no leftover weight-norm parts ---- */
    int leftovers = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *nm = gguf_get_tensor_name(gg, i);
        size_t len = strlen(nm);
        if ((len > 9 && !strcmp(nm + len - 9, ".weight_g")) ||
            (len > 9 && !strcmp(nm + len - 9, ".weight_v"))) leftovers++;
    }
    CHECK(leftovers == 0, "no .weight_g / .weight_v leftovers");

    /* ---- key tensors exist ---- */
    const char *keynames[] = {
        "emb_g.weight",
        "dec.conv_pre.weight",
        "dec.ups.0.weight",
        "flow.flows.0.enc.in_layers.0.weight",
        "enc_p.encoder.attn_layers.0.conv_q.weight",
    };
    for (size_t i = 0; i < sizeof(keynames)/sizeof(keynames[0]); i++)
        CHECK(find_tensor(gg, keynames[i]) >= 0, keynames[i]);

    /* ---- value/shape cross-checks against the .pth ---- */
    if (pth_path) {
        printf("== value cross-check vs %s ==\n", pth_path);
        pth_model_t *m = malloc(sizeof(pth_model_t)); // heap: 1.4 MB, not stack
        if (pth_load(m, pth_path) != 0) {
            fprintf(stderr, "  FAIL: cannot load pth for cross-check\n");
            free(m); gguf_free(gg); if (gctx) ggml_free(gctx);
            return g_fails + 1;
        }

        /* shapes == reversed pth shapes (regular + fused) */
        check_shape(gg, m, "emb_g.weight", 0);
        check_shape(gg, m, "dec.conv_pre.weight", 0);
        check_shape(gg, m, "dec.ups.0.weight", 1);
        check_shape(gg, m, "flow.flows.0.enc.in_layers.0.weight", 1);
        check_shape(gg, m, "enc_p.encoder.attn_layers.0.conv_q.weight", 0);

        /* fused weight-norm value */
        check_fused_value(gctx, m, "dec.ups.0.weight_v");

        /* regular tensor data identity */
        check_data_identity(gctx, m, "emb_g.weight");
        check_data_identity(gctx, m, "enc_p.encoder.attn_layers.0.conv_q.weight");

        pth_free(m);
        free(m);
    }

    printf("== %d/%d checks passed ==\n", g_checks - g_fails, g_checks);
    gguf_free(gg);
    if (gctx) ggml_free(gctx);
    return g_fails;
}

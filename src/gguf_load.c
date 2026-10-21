// gguf_load.c — load a GGUF RVC weight file into a ggml backend (CUDA/CPU).
// See gguf_load.h for the contract.
//
// Weights are loaded into a persistent ggml context and allocated on the
// compute backend. Each forward pass builds its graph in a fresh compute
// context and runs it through a backend scheduler (compute backend first,
// CPU last — required by the ggml scheduler).

#include "gguf_load.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPCV_MAX_INPUTS 24
#define SPCV_CTX_META_MB 512   // metadata buffer for the compute context

typedef struct {
    char name[256];
    const void *data;
    size_t nbytes;
} input_t;

struct spcv_ggml {
    ggml_backend_t be;        // compute backend (cuda or cpu)
    ggml_backend_t cpu;       // cpu backend (last in sched)
    ggml_backend_sched_t sched;
    ggml_backend_buffer_t wbuf; // weights buffer on the backend
    struct ggml_context *wctx;   // weights
    struct gguf_context *gguf;   // for name/shape lookup
    int be_id;                 // 0=cpu, 1=cuda
    char gguf_path[512];       // path to the .gguf file (for CPU re-read)
    // current forward pass
    struct ggml_context *ctx;    // compute context (current)
    struct ggml_cgraph *gf;      // compute graph
    struct ggml_context *oldctx; // previous compute context (free after sched_reset)
    input_t inputs[SPCV_MAX_INPUTS];
    int n_inputs;
};

/** Load weights into a backend and set up the scheduler. */
spcv_ggml_t * spcv_ggml_load(const char *path, int be, int dev) {
    spcv_ggml_t *g = (spcv_ggml_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    strncpy(g->gguf_path, path, sizeof(g->gguf_path) - 1);

    // 1. Metadata context (no_alloc=true: tensors created, no data yet).
    struct gguf_init_params gp = { /*no_alloc=*/true, &g->wctx };
    g->gguf = gguf_init_from_file(path, gp);
    if (!g->gguf || !g->wctx) {
        fprintf(stderr, "[gguf] cannot open %s\n", path);
        if (g->gguf) gguf_free(g->gguf);
        if (g->wctx) ggml_free(g->wctx);
        free(g); return NULL;
    }

    // 2. Pick the compute backend.
    int want_cuda = (be == SPCV_BE_CUDA) || (be == SPCV_BE_AUTO);
    g->be = NULL;
    if (want_cuda) {
        int n = ggml_backend_cuda_get_device_count();
        if (n > 0) {
            if (dev >= n) dev = 0;
            g->be = ggml_backend_cuda_init(dev);
            if (g->be) g->be_id = 1;
        }
    }
    if (!g->be) {
        if (be == SPCV_BE_CUDA) { fprintf(stderr, "[gguf] CUDA requested but unavailable\n"); }
        g->be = ggml_backend_cpu_init();
        if (!g->be) { fprintf(stderr, "[gguf] cpu backend init failed\n"); gguf_free(g->gguf); ggml_free(g->wctx); free(g); return NULL; }
        g->be_id = 0;
    }

    // 3. Allocate the (unallocated) weight tensors on the compute backend.
    g->wbuf = ggml_backend_alloc_ctx_tensors(g->wctx, g->be);
    if (!g->wbuf) {
        fprintf(stderr, "[gguf] alloc_ctx_tensors failed\n");
        ggml_backend_free(g->be); gguf_free(g->gguf); ggml_free(g->wctx); free(g); return NULL;
    }

    // 4. Copy each tensor's data from the GGUF file into the backend buffer.
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "[gguf] cannot reopen %s\n", path); ggml_backend_buffer_free(g->wbuf); ggml_backend_free(g->be); gguf_free(g->gguf); ggml_free(g->wctx); free(g); return NULL; }
    size_t data_off = gguf_get_data_offset(g->gguf);
    int copy_fail = 0;
    for (struct ggml_tensor *t = ggml_get_first_tensor(g->wctx); t; t = ggml_get_next_tensor(g->wctx, t)) {
        int64_t tid = gguf_find_tensor(g->gguf, ggml_get_name(t));
        if (tid < 0) { fprintf(stderr, "[gguf] tensor not in gguf: %s\n", ggml_get_name(t)); copy_fail = 1; break; }
        size_t off = gguf_get_tensor_offset(g->gguf, tid);
        size_t nbytes = ggml_nbytes(t);
        unsigned char *tbuf = (unsigned char *)malloc(nbytes ? nbytes : 1);
        if (fseek(fp, (long)(data_off + off), SEEK_SET) != 0 ||
            fread(tbuf, 1, nbytes, fp) != nbytes) {
            fprintf(stderr, "[gguf] read failed for %s\n", ggml_get_name(t));
            free(tbuf); copy_fail = 1; break;
        }
        ggml_backend_tensor_set(t, tbuf, 0, nbytes);
        free(tbuf);
    }
    fclose(fp);
    if (copy_fail) { ggml_backend_buffer_free(g->wbuf); ggml_backend_free(g->be); gguf_free(g->gguf); ggml_free(g->wctx); free(g); return NULL; }

    // 5. CPU backend (required as the LAST entry of the scheduler).
    g->cpu = ggml_backend_cpu_init();
    if (!g->cpu) { fprintf(stderr, "[gguf] cpu backend init failed\n"); ggml_backend_buffer_free(g->wbuf); ggml_backend_free(g->be); gguf_free(g->gguf); ggml_free(g->wctx); free(g); return NULL; }

    // 6. Scheduler: compute backend first, CPU last.
    ggml_backend_t backends[2] = { g->be, g->cpu };
    g->sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (!g->sched) { fprintf(stderr, "[gguf] sched_new failed\n"); ggml_backend_free(g->cpu); ggml_backend_free(g->be); ggml_backend_buffer_free(g->wbuf); gguf_free(g->gguf); ggml_free(g->wctx); free(g); return NULL; }

    printf("[gguf] loaded %s: %lld tensors on %s (dev %d)\n",
           path, (long long)gguf_get_n_tensors(g->gguf),
           g->be_id == 1 ? "CUDA" : "CPU", dev);
    return g;
}

struct ggml_tensor * spcv_ggml_tensor(const spcv_ggml_t *g, const char *name) {
    if (!g || !g->wctx) return NULL;
    return ggml_get_tensor(g->wctx, name);
}

/** Create a fresh compute context + graph for one forward pass. */
struct ggml_context * spcv_ggml_ctx(spcv_ggml_t *g) {
    if (!g) return NULL;
    // The previous compute context is kept alive until spcv_ggml_compute()
    // calls sched_reset (which releases the scheduler's reference to it).
    // We do NOT free it here — that would be a use-after-free.
    g->n_inputs = 0;
    struct ggml_init_params ip = { (size_t)SPCV_CTX_META_MB * 1024 * 1024, NULL, /*no_alloc=*/true };
    g->ctx = ggml_init(ip);
    if (!g->ctx) { fprintf(stderr, "[gguf] ggml_init(ctx) failed\n"); return NULL; }
    g->gf = ggml_new_graph(g->ctx);
    return g->ctx;
}

/** Register an input tensor (by name) with host data to upload before compute. */
void spcv_ggml_set_input(spcv_ggml_t *g, const char *name, const void *data, size_t nbytes) {
    if (!g || g->n_inputs >= SPCV_MAX_INPUTS) return;
    input_t *in = &g->inputs[g->n_inputs++];
    snprintf(in->name, sizeof(in->name), "%s", name);
    in->data = data; in->nbytes = nbytes;
}

enum ggml_status spcv_ggml_compute(spcv_ggml_t *g, struct ggml_tensor *out,
                                   void *dst, size_t nbytes) {
    if (!g || !g->ctx || !g->gf) return GGML_STATUS_FAILED;
    ggml_build_forward_expand(g->gf, out);
    // Reset the scheduler (releases the previous graph's allocation),
    // then free the previous compute context (now safe to free).
    ggml_backend_sched_reset(g->sched);
    if (g->oldctx) { ggml_free(g->oldctx); g->oldctx = NULL; }
    if (!ggml_backend_sched_alloc_graph(g->sched, g->gf)) return GGML_STATUS_FAILED;
    // Upload registered inputs (now allocated).
    for (int i = 0; i < g->n_inputs; i++) {
        struct ggml_tensor *t = ggml_graph_get_tensor(g->gf, g->inputs[i].name);
        if (t) ggml_backend_tensor_set(t, g->inputs[i].data, 0, g->inputs[i].nbytes);
        else fprintf(stderr, "[gguf] input not found: %s\n", g->inputs[i].name);
    }
    enum ggml_status st = ggml_backend_sched_graph_compute(g->sched, g->gf);
    if (st != GGML_STATUS_SUCCESS) return st;
    if (dst) ggml_backend_tensor_get(out, dst, 0, nbytes);
    // Keep the compute context alive (the scheduler still references it).
    // It will be freed by the next spcv_ggml_compute() or spcv_ggml_free().
    g->oldctx = g->ctx; g->ctx = NULL; g->gf = NULL; g->n_inputs = 0;
    return GGML_STATUS_SUCCESS;
}

int spcv_ggml_n_tensors(const spcv_ggml_t *g) {
    return g ? (int)gguf_get_n_tensors(g->gguf) : 0;
}
int spcv_ggml_backend_id(const spcv_ggml_t *g) { return g ? g->be_id : -1; }
const char * spcv_ggml_gguf_path(const spcv_ggml_t *g) { return g ? g->gguf_path : NULL; }

void spcv_ggml_free(spcv_ggml_t *g) {
    if (!g) return;
    if (g->ctx) ggml_free(g->ctx);
    if (g->oldctx) ggml_free(g->oldctx);
    if (g->sched) ggml_backend_sched_free(g->sched);
    if (g->wbuf) ggml_backend_buffer_free(g->wbuf);
    if (g->cpu) ggml_backend_free(g->cpu);
    if (g->be) ggml_backend_free(g->be);
    if (g->gguf) gguf_free(g->gguf);
    if (g->wctx) ggml_free(g->wctx);
    free(g);
}

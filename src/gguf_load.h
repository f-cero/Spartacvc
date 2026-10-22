// gguf_load.h — load a GGUF RVC weight file into a ggml backend (CUDA/CPU).
//
// Wraps the "load weights into a backend + provide a scheduler for compute"
// plumbing so the graph-building code (ggml_rvc.c) can focus on the math.
//
// Model:
//   - Weights live in a persistent ggml context (`wctx`) whose tensors are
//     allocated on the compute backend and filled from the GGUF file.
//   - Each forward pass builds its graph in a fresh compute context (`ctx`),
//     then runs it through a backend scheduler (compute backend first, CPU last).
//
// GGML LAYOUT CONVENTIONS (verified 2026-07-21, test_rvc_conv):
//   * Tensors are row-major with ne[0] FASTEST (contiguous): element(i,j) at
//     j*nb[1] + i*nb[0]. PyTorch has the last dim fastest, but for a flat byte
//     copy the fastest dim is the same on both sides, so the .pth->GGUF flat
//     copy is valid:
//       Linear   [out,in]   -> [in,out]   (in fastest on both)
//       Conv1d   [out,in,K] -> [K,in,out] (K fastest on both)
//       ConvTr   [in,out,K] -> [K,out,in] (K fastest on both)
//   * conv_1d: data [L, IC] (ne[0]=L, ne[1]=IC); output [OL, OC]. Padding
//     p0=K/2 (symmetric, matches nn.Conv1d). Bias is NOT applied - add it as
//     a [1, OC] tensor broadcast over time.
//   * mul_mat(a, b): a [k,n], b [k,m] -> [n,m] (reduction over ne[0]).
//   * soft_max sums over ne[0].
#ifndef SPCV_GGUF_LOAD_H
#define SPCV_GGUF_LOAD_H

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend selection for spcv_ggml_load().
enum {
    SPCV_BE_CPU  = 0,   // CPU only
    SPCV_BE_CUDA = 1,   // NVIDIA CUDA
    SPCV_BE_AUTO = 2    // CUDA if available, else CPU
};

// Opaque handle.
typedef struct spcv_ggml spcv_ggml_t;

/**
 * Load a GGUF RVC weight file into a ggml backend.
 * @param path  Path to the .gguf file.
 * @param be    SPCV_BE_CPU / SPCV_BE_CUDA / SPCV_BE_AUTO.
 * @param dev   Device index (used for CUDA; ignored for CPU).
 * @return      A handle, or NULL on failure (message on stderr).
 */
spcv_ggml_t * spcv_ggml_load(const char *path, int be, int dev);

/**
 * Get a weight tensor by name (e.g. "dec.conv_pre.weight").
 * @return the tensor, or NULL if not present.
 */
struct ggml_tensor * spcv_ggml_tensor(const spcv_ggml_t *g, const char *name);

/**
 * Return a fresh compute context for building one forward-pass graph.
 * The caller creates tensors in it and calls spcv_ggml_compute() with the
 * output tensor. The context is freed automatically after compute.
 */
struct ggml_context * spcv_ggml_ctx(spcv_ggml_t *g);

/**
 * Register an input leaf (by name) with host data to upload before compute.
 * Must be called after spcv_ggml_ctx() and before spcv_ggml_compute().
 */
void spcv_ggml_set_input(spcv_ggml_t *g, const char *name, const void *data, size_t nbytes);

/**
 * Build (expand) and compute a graph whose result is `out`.
 * Copies the result into the caller buffer pointed to by *out_data
 * (which must hold out->ne[0]*...*(size) bytes). Returns GGML_STATUS_SUCCESS.
 */
enum ggml_status spcv_ggml_compute(spcv_ggml_t *g, struct ggml_tensor *out,
                                   void *dst, size_t nbytes);

/** Read a tensor's raw data from the GGUF file into a host buffer.
 *  @return 0 on success, -1 if tensor not found or read failed. */
int spcv_ggml_get_tensor(const spcv_ggml_t *g, const char *name,
                         void *dst, size_t nbytes, int64_t *out_n);

/** Number of weight tensors (for diagnostics). */
int spcv_ggml_n_tensors(const spcv_ggml_t *g);

/** Which backend is active: 0=cpu, 1=cuda. */
int spcv_ggml_backend_id(const spcv_ggml_t *g);

/** Get the GGUF file path (for CPU re-read of weights). */
const char * spcv_ggml_gguf_path(const spcv_ggml_t *g);

/** Free the handle and all backends/contexts. */
void spcv_ggml_free(spcv_ggml_t *g);

#ifdef __cplusplus
}
#endif

#endif // SPCV_GGUF_LOAD_H

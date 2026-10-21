// pth2gguf.h — in-app, in-memory .pth → GGUF conversion.
//
// REQUIREMENT (docs/pth2gguf-in-app-2026-07-21.md): the app itself converts
// .pth voice models to GGUF in memory so the ggml/CUDA runtime can load them
// transparently. No temp files on disk, no external converter.
//
// SPEC reference: tools/pth-to-gguf.md (naming, weight-norm fusion, shapes).
// The shapes match ggml's op layouts exactly:
//   Linear          [out,in]     -> [in,out]      (ggml_mul_mat)
//   Conv1d          [out,in,K]   -> [K,in,out]    (ggml_conv_1d)
//   ConvTranspose1d [in,out,K]   -> [K,out,in]    (ggml_conv_transpose_1d)
#ifndef SPARTACVC_PTH2GGUF_H
#define SPARTACVC_PTH2GGUF_H

#include "pth_loader.h"
#include "ggml.h"
#include "gguf.h"

// A converted RVC model: ggml tensor store + GGUF metadata/tensor table.
typedef struct {
    struct ggml_context * gctx;    // weights live here (must stay alive)
    struct gguf_context * ggctx;   // metadata + tensor table
} spcv_gguf_t;

/**
 * Convert a loaded .pth model into an in-memory GGUF.
 *  - normalizes tensor names (strips module./model./net_g./synthesizer./generator.)
 *  - fuses weight-norm pairs (foo.weight_g + foo.weight_v -> foo.weight)
 *  - stores tensors with reversed dims (ggml layout, flat byte copy)
 *  - sets rvc.* metadata from the parsed config
 * Returns 0 on success, -1 on error (and fills *out with nothing).
 */
int spcv_pth2gguf(const pth_model_t *m, spcv_gguf_t *out);

/** Release the ggml context and gguf context. Safe on zeroed struct. */
void spcv_gguf_free(spcv_gguf_t *m);

/** Look up a tensor by (normalized) name. Returns the ggml tensor or NULL. */
struct ggml_tensor * spcv_gguf_tensor(const spcv_gguf_t *m, const char *name);

#endif

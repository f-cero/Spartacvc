// gguf_check.h — verification of converted .gguf RVC weight files.
//
// Used by:  spartacvc --check-gguf model.gguf [--voice model.pth]
//
// Structural checks (always):
//   - file opens, has tensors
//   - metadata (general.architecture == rvc, rvc.weight_norm == fused)
//   - no leftover .weight_g / .weight_v tensors
//   - key tensors exist with the expected (ggml-layout) shapes
//
// Value checks (when a .pth is also given):
//   - fused weight-norm values match  W = V * G / ||V||  (recomputed from .pth)
//   - regular tensor data is byte-identical to the .pth (flat copy)
#ifndef SPARTACVC_GGUF_CHECK_H
#define SPARTACVC_GGUF_CHECK_H

/**
 * @brief  Verify a .gguf weight file.
 * @param  gguf_path  path to the .gguf to check
 * @param  pth_path   optional raw .pth for value cross-checks (NULL = structure only)
 * @return number of failed checks (0 = all pass)
 */
int spcv_gguf_check(const char *gguf_path, const char *pth_path);

#endif

// ggml_rvc_enc.h — RVC v2 encoder on ggml backend.
#ifndef SPCV_GGML_RVC_ENC_H
#define SPCV_GGML_RVC_ENC_H

#include "gguf_load.h"

/**
 * Run the full RVC v2 encoder (embedding + 6 attn/FFN layers + proj).
 * @param g        ggml handle with loaded weights.
 * @param phone    Phone embeddings [T, Q] (time-major, Q fastest per row).
 * @param T        Sequence length.
 * @param pitch    Pitch indices [T] (int64, 0-255).
 * @param m_p      Output mean [FC, T] (C-major: c*T + t).
 * @param logs_p   Output log-std [FC, T].
 * @param P        Hidden channels (192).
 * @param Q        Embedding dim (768).
 * @param FF       FFN hidden (768).
 * @param H        Num heads (2).
 * @param NL       Num layers (6).
 * @param FK       FFN kernel (3).
 * @param FC       Flow channels (192).
 * @param max_rel  Max relative positions (21).
 * @param hd       Head dim (96).
 * @return 0 on success.
 */
int spcv_ggml_encoder(spcv_ggml_t *g,
                      const float *phone, int T,
                      const int64_t *pitch,
                      float *m_p, float *logs_p,
                      int P, int Q, int FF, int H, int NL, int FK, int FC,
                      int max_rel, int hd);

#endif

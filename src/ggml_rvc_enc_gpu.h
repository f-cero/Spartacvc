/**
 * @file    ggml_rvc_enc_gpu.h
 * @brief   RVC v2 encoder via ggml GPU ops (mul_mat, norm, conv_1d, relu).
 *
 * Runs the 6-layer transformer encoder on the ggml backend (CUDA/CPU).
 * Attention (relative positions) is computed on host between GPU steps.
 *
 * Layout convention:
 *   - Internal ggml tensors: [C, T] (ne[0]=C fastest) — C-fast
 *   - Output m_p/logs_p: [C, T] C-fast (m_p[c + t*C]) to match flow input
 */
#ifndef SPCV_GGML_RVC_ENC_GPU_H
#define SPCV_GGML_RVC_ENC_GPU_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Run the RVC v2 encoder via ggml GPU ops.
 *
 * @param  gguf_path  path to the .gguf voice model
 * @param  be_id      0=CPU, 1=CUDA
 * @param  gpu_id     device index
 * @param  phone      one-hot phone input [emb_dim, T] C-fast (phone[c + t*emb_dim])
 * @param  emb_dim    phone embedding dimension (768)
 * @param  T          number of frames
 * @param  pitch      pitch class indices [T] (int64, 0..255)
 * @param  m_p        [out] mean [192, T] T-fast (m_p[c*T + t])
 * @param  logs_p     [out] log-sigma [192, T] T-fast (logs_p[c*T + t])
 * @return 0 on success, -1 on failure
 */
int spcv_ggml_encoder(const char *gguf_path, int be_id, int gpu_id,
                      const float *phone, int emb_dim, int T,
                      const int64_t *pitch,
                      float *m_p, float *logs_p);

#ifdef __cplusplus
}
#endif
#endif

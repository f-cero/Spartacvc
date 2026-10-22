/**
 * @file    ggml_pipeline.h
 * @brief   ggml-based RVC inference pipeline (encoder → flow → decoder on GPU).
 *
 * Replaces the pure-C rvc_infer() with ggml ops on CUDA/Vulkan/CPU.
 * HuBERT/RMVPE remain ONNX (CPU). Only the voice model runs through ggml.
 */
#ifndef SPCV_GGML_PIPELINE_H
#define SPCV_GGML_PIPELINE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Run the full RVC pipeline (encoder+flow+decoder) via ggml.
 *
 * @param  gguf_path  path to the .gguf voice model
 * @param  be_id      0=CPU, 1=CUDA, 2=Vulkan
 * @param  gpu_id     device index
 * @param  emb        HuBERT features [emb_dim, T] C-fast (dim fastest)
 * @param  emb_dim    embedding dimension (768)
 * @param  T          number of frames
 * @param  pitch      pitch values [T] (int64, RVC scale 1..255)
 * @param  f0         f0 values [T] (Hz)
 * @param  out        [out] output PCM buffer (caller allocates T*480 floats)
 * @param  out_len    [out] number of output samples
 * @return 0 on success, -1 on failure
 */
int spcv_ggml_infer(const char *gguf_path, int be_id, int gpu_id,
                    const float *emb, int emb_dim, int T,
                    const int64_t *pitch, const float *f0,
                    float *out, int *out_len);

/**
 * @brief  Run flow+decoder via ggml (encoder output provided externally).
 *
 * @param  gguf_path  path to the .gguf voice model
 * @param  be_id      0=CPU, 1=CUDA, 2=Vulkan
 * @param  gpu_id     device index
 * @param  m_p        encoder mean [192, T] C-fast
 * @param  logs_p     encoder log-sigma [192, T] C-fast
 * @param  g_emb      speaker embedding [256]
 * @param  T          number of frames
 * @param  out        [out] output PCM (caller allocates T*480 floats)
 * @param  out_len    [out] number of output samples
 * @return 0 on success, -1 on failure
 */
int spcv_ggml_flow_dec(const char *gguf_path, int be_id, int gpu_id,
                       const float *m_p, const float *logs_p,
                       const float *g_emb, int T,
                       const float *f0,   /* f0 [T] Hz, for SineGen */
                       float *out, int *out_len);

#ifdef __cplusplus
}
#endif
#endif

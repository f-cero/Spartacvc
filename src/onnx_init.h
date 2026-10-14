/**
 * @file    onnx_init.h
 * @brief   Read ONNX graph initializers to extract model config values.
 *
 * RVC voice models bake their hyperparameters (sr, upsample_rates, etc.)
 * into the ONNX graph as initializer tensors.  We scan the initializers
 * to find the values we need to compute the model's native frame rate
 * and output sample rate.
 *
 * This avoids any hardcoded constants — every model reports its own
 * parameters.
 */
#ifndef SPCV_ONNX_INIT_H
#define SPCV_ONNX_INIT_H

#include <stdint.h>
#include "onnx.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Extract model config from a voice model's ONNX initializers.
 *
 * Scans the session's initializers for known RVC config tensors and
 * derives the output sample rate and frame rate.
 *
 * @param  sess       loaded voice model session
 * @param  sr_out     [out] output sample rate
 * @param  fps_out    [out] native frame rate (sr / prod(upsample_rates))
 * @param  dim_out    [out] phone embedding dimension (256 or 768)
 * @return  0 on success, -1 if config could not be determined
 *
 * Strategy:
 *   - Check the phone input shape: dim is the last axis
 *   - Infer sr from the decoder's output: the voice model's output
 *     sample rate is encoded in the upsampling network.  For RVC v2
 *     models the standard rates are 32000 or 48000.
 *   - If we can read the upsample_rates from initializers, compute
 *     fps = sr / prod(upsample_rates).
 *   - Fallback: use the phone dimension to pick the known config
 *     (256-dim → 32kHz/400 = 80fps, 768-dim → 48kHz/480 = 100fps).
 */
int spcv_voice_model_config(spcv_onnx_session *sess,
                            int *sr_out, int *fps_out, int *dim_out);

#ifdef __cplusplus
}
#endif

#endif /* SPCV_ONNX_INIT_H */

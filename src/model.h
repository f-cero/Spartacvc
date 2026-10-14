/**
 * @file    model.h
 * @brief   Per-model parameters derived from the voice model's config.
 *
 * RVC voice models each have their own sample rate and upsample rates,
 * which determine:
 *   - output sample rate  = config sr
 *   - native frame rate   = sr / prod(upsample_rates)
 *
 * These values MUST be read from the model, never hardcoded.
 */
#ifndef SPCV_MODEL_H
#define SPCV_MODEL_H

#include <stdint.h>

/**
 * @brief  Parameters extracted from a voice model's config.
 */
typedef struct {
    int   sr;            /**< output sample rate (e.g. 32000, 48000)   */
    int   upsample_prod; /**< product of upsample_rates (e.g. 400, 480)*/
    int   fps;           /**< native frame rate = sr / upsample_prod   */
    int   phone_dim;     /**< HuBERT embedding dim (256 or 768)        */
} spcv_model_params;

/**
 * @brief  Compute model parameters from raw config values.
 *
 * @param  sr             sample rate from config
 * @param  upsample_rates array of upsample rate integers
 * @param  n_rates        number of upsample rates
 * @param  phone_dim      embedding dimension (256 or 768)
 * @param  out            [out] computed parameters
 */
void spcv_model_params_from_config(int sr, const int *upsample_rates,
                                   int n_rates, int phone_dim,
                                   spcv_model_params *out);

#ifdef __cplusplus
}
#endif

#endif /* SPCV_MODEL_H */

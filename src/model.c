/**
 * @file    model.c
 * @brief   Per-model parameter computation.
 */
#include "model.h"
#include "debug.h"

void spcv_model_params_from_config(int sr, const int *upsample_rates,
                                   int n_rates, int phone_dim,
                                   spcv_model_params *out) {
    out->sr = sr;
    out->upsample_prod = 1;
    for (int i = 0; i < n_rates; i++)
        out->upsample_prod *= upsample_rates[i];
    out->fps = sr / out->upsample_prod;
    out->phone_dim = phone_dim;
    SPCV_TRACE("model", "sr=%d upsample_prod=%d fps=%d dim=%d",
               out->sr, out->upsample_prod, out->fps, out->phone_dim);
}

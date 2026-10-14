/**
 * @file    f0.h
 * @brief   Pitch (f0) extraction via RMVPE + cents decoding + pitch shift.
 *
 * Pipeline role:
 *   mel[time, 128]  ->  RMVPE ONNX  ->  sal[time, 360]
 *   sal  ->  cents decode (9-bin avg)  ->  f0 [Hz] per frame
 *   f0  ->  pitch shift (semitones)  ->  shifted f0
 *   f0  ->  quantize to 1..255 (RVC pitch input)
 *
 * RMVPE expects mel transposed: [1, 128, time] (channels-first)
 * RMVPE requires time dim to be a multiple of 32 (UNet constraint).
 */
#ifndef SPCV_F0_H
#define SPCV_F0_H

#include <stdint.h>
#include "onnx.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Extract f0 from a mel spectrogram using RMVPE.
 *
 * @param  mel       mel spectrogram [n_frames, 128] (log-mel, row-major)
 * @param  n_frames  number of time frames
 * @param  rmvpe     loaded RMVPE session
 * @param  f0_out    [out] f0 in Hz, one value per frame (caller allocates)
 * @return  0 on success, -1 on failure
 *
 * Unvoiced frames have f0 = 0.
 */
int spcv_f0_extract(const float *mel, int n_frames,
                    spcv_onnx_session *rmvpe, float *f0_out);

/**
 * @brief  Apply pitch shift in semitones.
 *
 * @param  f0        [in/out] f0 array (modified in place)
 * @param  n_frames  length
 * @param  semitones  shift amount (e.g. +12 = one octave up)
 *
 * Unvoiced frames (f0 == 0) are left unchanged.
 */
void spcv_f0_shift(float *f0, int n_frames, int semitones);

/**
 * @brief  Quantize f0 to RVC's 1..255 pitch scale.
 *
 * @param  f0        f0 in Hz
 * @param  n_frames  length
 * @param  pitch_out [out] int64 array, values 1..255 (caller allocates)
 *
 * Mapping: log-f0 scaled to [50 Hz, 1100 Hz] -> [1, 255]
 * Unvoiced (f0 == 0) maps to 1.
 */
void spcv_f0_quantize(const float *f0, int n_frames, int64_t *pitch_out);

#ifdef __cplusplus
}
#endif

#endif /* SPCV_F0_H */

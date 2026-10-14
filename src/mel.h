/**
 * @file    mel.h
 * @brief   Mel spectrogram computation (hand-rolled, no external deps).
 *
 * Pipeline role:
 *   16 kHz audio  ->  mel[time, 128]  ->  RMVPE  ->  f0
 *
 * Parameters (RVC standard):
 *   n_fft   = 1024
 *   hop     = 160
 *   n_mels  = 128
 *   fmin    = 30 Hz
 *   fmax    = 8000 Hz
 *   sr      = 16000 Hz
 */
#ifndef SPCV_MEL_H
#define SPCV_MEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Compute a mel spectrogram from 16 kHz mono audio.
 *
 * @param  audio     input samples (16 kHz, mono, float32 in [-1, 1])
 * @param  n_samples number of samples
 * @param  mel_out   [out] mel spectrogram buffer (caller allocates)
 * @param  n_mels    number of mel bins (typically 128)
 * @return  number of time frames, or -1 on error
 *
 * The output is log-mel: log(clip(S @ mel_fb.T, 1e-5, inf))
 * Shape: [n_frames, n_mels]
 *
 * Complexity: O(n_samples * n_fft / 2 + n_frames * n_mels * n_fft / 2)
 */
int spcv_mel_compute(const float *audio, int64_t n_samples,
                     float *mel_out, int n_mels);

/**
 * @brief  Number of time frames for a given number of 16 kHz samples.
 * @param  n_samples  input sample count
 * @return  frame count (at least 1 if n_samples > 0)
 */
int64_t spcv_mel_n_frames(int64_t n_samples);

#ifdef __cplusplus
}
#endif

#endif /* SPCV_MEL_H */

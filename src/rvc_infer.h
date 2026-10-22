#ifndef SPARTACVC_RVC_INFER_H
#define SPARTACVC_RVC_INFER_H
#include "rvc.h"
int rvc_infer(const rvc_model_t *m,
              const float *phone, int T,
              const int64_t *pitch, const float *pitchf,
              const float *noise,
              float *out, int *out_len,
              uint32_t *seed);

/**
 * @brief  Encoder forward only (for testing / ggml comparison).
 *         phone[T,ques] + pitch[T] -> m_p[FC,T], logs_p[FC,T] (C-major).
 */
int rvc_encoder_forward(const rvc_model_t *m,
                        const float *phone, int T,
                        const int64_t *pitch,
                        float *m_p, float *logs_p);
#endif

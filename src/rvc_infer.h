#ifndef SPARTACVC_RVC_INFER_H
#define SPARTACVC_RVC_INFER_H
#include "rvc.h"
int rvc_infer(const rvc_model_t *m,
              const float *phone, int T,
              const int64_t *pitch, const float *pitchf,
              const float *noise,
              float *out, int *out_len,
              uint32_t *seed);
#endif

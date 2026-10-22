// Generate pure-C encoder reference output
#include "rvc.h"
#include "rvc_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FC 192
#define Q_DIM 768
#define T_LEN 16

int main() {
    float *phone = malloc((size_t)T_LEN * Q_DIM * 4);
    int64_t *pitch = malloc(T_LEN * 8);
    srand(42);
    for (int i = 0; i < T_LEN * Q_DIM; i++) phone[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int t = 0; t < T_LEN; t++) pitch[t] = (t * 37) % 256;

    rvc_model_t *m = (rvc_model_t *)calloc(1, sizeof(rvc_model_t));
    if (rvc_load(m, "models/voices/nahida-jp.pth") != 0) { fprintf(stderr, "load fail\n"); return 2; }
    float *m_p = malloc((size_t)FC * T_LEN * 4);
    float *logs_p = malloc((size_t)FC * T_LEN * 4);
    rvc_encoder_forward(m, phone, T_LEN, pitch, m_p, logs_p);

    FILE *f = fopen("scratch/enc_ref_m_p.bin", "wb");
    fwrite(m_p, sizeof(float), FC * T_LEN, f);
    fclose(f);
    f = fopen("scratch/enc_ref_logs_p.bin", "wb");
    fwrite(logs_p, sizeof(float), FC * T_LEN, f);
    fclose(f);
    printf("saved ref: m_p[0..5]=%g %g %g %g %g %g\n", m_p[0],m_p[1],m_p[2],m_p[3],m_p[4],m_p[5]);
    printf("saved ref: logs_p[0..5]=%g %g %g %g %g %g\n", logs_p[0],logs_p[1],logs_p[2],logs_p[3],logs_p[4],logs_p[5]);

    free(phone); free(pitch); free(m_p); free(logs_p);
    rvc_free(m); free(m);
    return 0;
}

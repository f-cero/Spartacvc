// test_pth.c — test .pth loading with the Nahida model
#define _CRT_SECURE_NO_WARNINGS
#include "pth_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
    const char *path = (argc > 1) ? argv[1] : "C:/temp/nahida_extracted/nahida-jp.pth";

    printf("Loading: %s\n\n", path);

    pth_model_t *m = (pth_model_t *)calloc(1, sizeof(pth_model_t));
    if (!m) { fprintf(stderr, "OOM\n"); return 1; }

    if (pth_load(m, path) != 0) {
        fprintf(stderr, "FAILED to load .pth\n");
        free(m);
        return 1;
    }

    pth_print_summary(m);

    // Test: read a specific tensor and verify values
    printf("\n--- Tensor read test ---\n");
    const pth_tensor_t *t = pth_find_tensor(m, "enc_p.emb_phone.weight");
    if (t) {
        int64_t n = 1;
        for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
        printf("Reading %s: %lld elements\n", t->name, (long long)n);

        float *buf = (float *)malloc(n * sizeof(float));
        if (pth_read_tensor_f32(m, t, buf) == 0) {
            printf("  First 8 values:");
            for (int i = 0; i < 8 && i < n; i++)
                printf(" %.5f", buf[i]);
            printf("\n");
            // Check for NaN/Inf
            int bad = 0;
            for (int64_t i = 0; i < n; i++) {
                if (buf[i] != buf[i] || buf[i] == INFINITY || buf[i] == -INFINITY) bad++;
            }
            printf("  NaN/Inf count: %d / %lld\n", bad, (long long)n);
            free(buf);
        } else {
            printf("  FAILED to read tensor\n");
        }
    } else {
        printf("  Tensor 'enc_p.emb_phone.weight' not found!\n");
        // List available tensors
        for (int i = 0; i < m->pkl.n_tensors && i < 5; i++)
            printf("  available: %s\n", m->pkl.tensors[i].name);
    }

    // Verify config
    if (m->pkl.has_config) {
        pth_config_t *c = &m->pkl.config;
        int prod = 1;
        for (int i = 0; i < c->n_upsample; i++) prod *= c->upsample_rates[i];
        printf("\n--- Config verification ---\n");
        printf("  sr=%d, prod(upsample)=%d, frame_rate=%d (expected %d)\n",
               c->sr, prod, c->frame_rate, c->sr / prod);
        printf("  ques=%d (768=768dim model, 256=256dim model)\n", c->ques);
        printf("  n_mel=%d (expected 128 for RMVPE)\n", c->n_mel);

        int ok = (c->sr == 48000) && (c->frame_rate == 100) && (c->ques == 768);
        printf("  %s\n", ok ? "PASS ✓" : "FAIL ✗");
    }

    pth_free(m);
    free(m);
    printf("\nDone.\n");
    return 0;
}

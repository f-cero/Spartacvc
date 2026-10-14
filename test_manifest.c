// test_manifest.c — verify manifest loading
#include "src/manifest.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "C:/temp/nahida.spacvc";
    
    manifest_t *m = calloc(1, sizeof(manifest_t));
    if (!m) { fprintf(stderr, "OOM manifest\n"); return 1; }
    
    if (manifest_load(m, dir) != 0) {
        fprintf(stderr, "FAILED to load manifest\n");
        free(m);
        return 1;
    }

    printf("Manifest loaded: %s\n", dir);
    printf("Tensors: %d\n", m->n_tensors);
    printf("Config: %d values\n", m->config.count);
    for (int i = 0; i < m->config.count; i++)
        printf("  [%d] = %lld\n", i, (long long)m->config.values[i]);

    printf("\nFirst 10 tensors:\n");
    for (int i = 0; i < 10 && i < m->n_tensors; i++) {
        manifest_tensor_t *t = &m->tensors[i];
        printf("  [%d] %s: ", i, t->name);
        for (int d = 0; d < t->ndim; d++) printf("%lld%s", (long long)t->shape[d], d < t->ndim-1 ? "x" : "");
        printf("  dtype=%d file=%d size=%llu\n", t->dtype, t->data_file, (unsigned long long)t->size_bytes);
    }

    // Load first tensor and check values
    printf("\nLoading tensor[0] (%s)...\n", m->tensors[0].name);
    manifest_tensor_t *t0 = &m->tensors[0];
    size_t nbytes = (size_t)t0->size_bytes;
    uint8_t *buf = malloc(nbytes);
    if (!buf) { fprintf(stderr, "OOM tensor buf\n"); free(m); return 1; }

    if (manifest_load_tensor(m, 0, buf) != 0) {
        fprintf(stderr, "FAILED to load tensor data\n");
        free(buf); free(m);
        return 1;
    }

    if (t0->dtype == 0) { // fp16
        printf("First 8 fp16 values: ");
        for (int i = 0; i < 8; i++) {
            uint16_t h = buf[2*i] | (buf[2*i+1] << 8);
            int sign = (h >> 15) & 1;
            int exp = (h >> 10) & 0x1F;
            int mant = h & 0x3FF;
            float v;
            if (exp == 0) v = ldexpf((float)mant, -14);
            else if (exp == 31) v = (mant ? NAN : INFINITY);
            else v = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
            if (sign) v = -v;
            printf("%s%.6f", i ? " " : "", v);
        }
        printf("\n");
    }
    free(buf);

    uint64_t total = 0;
    for (int i = 0; i < m->n_tensors; i++)
        total += m->tensors[i].size_bytes;
    printf("\nTotal tensor data: %llu bytes (%.1f MB)\n",
           (unsigned long long)total, (double)total / 1048576.0);

    printf("\nPASS\n");
    free(m);
    return 0;
}

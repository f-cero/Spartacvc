// pth_loader.c — load .pth files (ZIP + pickle) into C structures
#include "pth_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Simple fp16 → fp32 conversion
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(h & 0x3FF);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // subnormal
            int e = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return *(float *)&f;
}

int pth_load(pth_model_t *m, const char *path) {
    memset(m, 0, sizeof(*m));

    // 1. Open ZIP
    if (zip_open(&m->zip, path) != 0) {
        fprintf(stderr, "pth_load: failed to open ZIP: %s\n", path);
        return -1;
    }
    m->owns_zip = 1;

    // 2. Find data.pkl
    // The name varies: "nahida-jp_e102_s21114/data.pkl"
    // Scan for any entry ending in "data.pkl"
    const zip_entry_t *pkl_entry = NULL;
    for (int i = 0; i < m->zip.count; i++) {
        const char *name = m->zip.entries[i].name;
        size_t len = strlen(name);
        if (len >= 8 && strcmp(name + len - 8, "data.pkl") == 0) {
            pkl_entry = &m->zip.entries[i];
            break;
        }
    }
    if (!pkl_entry) {
        fprintf(stderr, "pth_load: data.pkl not found in ZIP\n");
        pth_free(m);
        return -1;
    }

    // 3. Read and parse pickle
    uint8_t *pkl_buf = zip_read_all(&m->zip, pkl_entry);
    if (!pkl_buf) {
        fprintf(stderr, "pth_load: failed to read data.pkl\n");
        pth_free(m);
        return -1;
    }

    int rc = pickle_parse(&m->pkl, pkl_buf, pkl_entry->size);
    free(pkl_buf);
    if (rc != 0) {
        fprintf(stderr, "pth_load: pickle parse error: %s\n", m->pkl.error_msg);
        pth_free(m);
        return -1;
    }

    // 4. Index data files (data/0, data/1, ...)
    for (int i = 0; i < m->zip.count; i++) {
        const char *name = m->zip.entries[i].name;
        // Look for "/data/N" pattern at the end
        size_t len = strlen(name);
        if (len < 6) continue;
        // Find last "/data/"
        const char *p = strrchr(name, '/');
        if (!p) continue;
        // Check if the directory part ends with "data"
        // Actually the pattern is: <prefix>/data/<N>
        // So the filename is just the number
        if (p == name) continue;
        // Check if char before the number is '/'
        // The entry name is like "nahida-jp_e102_s21114/data/0"
        // So we need to find "/data/" in the name
        const char *d = strstr(name, "/data/");
        if (!d) continue;
        const char *num_start = d + 6;
        int idx = atoi(num_start);
        if (idx >= 0 && idx < 1024) {
            m->data_files[idx] = &m->zip.entries[i];
            if (idx + 1 > m->n_data_files) m->n_data_files = idx + 1;
        }
    }

    return 0;
}

void pth_free(pth_model_t *m) {
    if (m->owns_zip && m->zip.fh) {
        zip_close(&m->zip);
        m->owns_zip = 0;
    }
    // Free pickle values
    for (int i = 0; i < PK_MAX_MEMO; i++) {
        if (m->pkl.memo[i]) {
            pkv_free(m->pkl.memo[i]);
            m->pkl.memo[i] = NULL;
        }
    }
    // Free stack
    for (int i = 0; i < m->pkl.sp; i++) {
        if (m->pkl.stack[i]) {
            pkv_free(m->pkl.stack[i]);
            m->pkl.stack[i] = NULL;
        }
    }
    memset(m, 0, sizeof(*m));
}

const pth_tensor_t *pth_find_tensor(const pth_model_t *m, const char *name) {
    for (int i = 0; i < m->pkl.n_tensors; i++) {
        if (strcmp(m->pkl.tensors[i].name, name) == 0)
            return &m->pkl.tensors[i];
    }
    return NULL;
}

int pth_read_tensor_raw(const pth_model_t *m, const pth_tensor_t *t, uint8_t *out) {
    if (t->data_file < 0 || t->data_file >= m->n_data_files) return -1;
    const zip_entry_t *entry = m->data_files[t->data_file];
    if (!entry) return -1;
    if (t->size_bytes > entry->size) return -1;
    return zip_read(&m->zip, entry, out, t->size_bytes);
}

int pth_read_tensor_f32(const pth_model_t *m, const pth_tensor_t *t, float *out) {
    // Calculate number of elements
    int64_t n_elem = 1;
    for (int i = 0; i < t->ndim; i++) n_elem *= t->shape[i];

    uint8_t *raw = (uint8_t *)malloc(t->size_bytes);
    if (!raw) return -1;
    if (pth_read_tensor_raw(m, t, raw) != 0) {
        free(raw);
        return -1;
    }

    switch (t->dtype) {
    case 0: // fp16
        for (int64_t i = 0; i < n_elem; i++) {
            uint16_t h = (uint16_t)(raw[i*2] | (raw[i*2+1] << 8));
            out[i] = fp16_to_f32(h);
        }
        break;
    case 1: // fp32
        memcpy(out, raw, n_elem * 4);
        break;
    case 2: // int32
        for (int64_t i = 0; i < n_elem; i++)
            out[i] = (float)((int32_t *)(raw))[i];
        break;
    case 3: // int64
        for (int64_t i = 0; i < n_elem; i++)
            out[i] = (float)((int64_t *)(raw))[i];
        break;
    default:
        free(raw);
        return -1;
    }

    free(raw);
    return 0;
}

void pth_print_summary(const pth_model_t *m) {
    printf("=== .pth Model Summary ===\n");
    printf("Tensors: %d\n", m->pkl.n_tensors);
    printf("Data files: %d\n", m->n_data_files);
    printf("ZIP entries: %d\n", m->zip.count);

    if (m->pkl.has_config) {
        pth_config_t *c = &m->pkl.config;
        printf("\n--- Config ---\n");
        printf("  n_mel:            %d\n", c->n_mel);
        printf("  filter_channels:  %d\n", c->filter_channels);
        printf("  ff_dim:           %d\n", c->ff_dim);
        printf("  p_dim:            %d\n", c->p_dim);
        printf("  ques (embed dim): %d\n", c->ques);
        printf("  p_kernel:         %d\n", c->p_kernel);
        printf("  p_heads:          %d\n", c->p_heads);
        printf("  p_layers:         %d\n", c->p_layers);
        printf("  upsample_rates:   ");
        for (int i = 0; i < c->n_upsample; i++) printf("%d ", c->upsample_rates[i]);
        printf("\n");
        printf("  upsample_init:    %d\n", c->upsample_initial);
        printf("  upsample_kernels: ");
        for (int i = 0; i < c->n_upsample; i++) printf("%d ", c->upsample_kernels[i]);
        printf("\n");
        printf("  resblock_kernel:  %d\n", c->resblock_kernel);
        printf("  resblock_ch:      %d\n", c->resblock_ch);
        printf("  sr:               %d\n", c->sr);
        printf("  frame_rate:       %d fps\n", c->frame_rate);
    }

    printf("\n--- First 10 tensors ---\n");
    for (int i = 0; i < m->pkl.n_tensors && i < 10; i++) {
        pth_tensor_t *t = &m->pkl.tensors[i];
        printf("  %s: shape=[", t->name);
        for (int d = 0; d < t->ndim; d++) printf("%lld%s", (long long)t->shape[d], d < t->ndim-1 ? "," : "");
        printf("] dtype=%d file=%d size=%lld\n",
               t->dtype, t->data_file, (long long)t->size_bytes);
    }
    if (m->pkl.n_tensors > 10)
        printf("  ... (%d more)\n", m->pkl.n_tensors - 10);
}

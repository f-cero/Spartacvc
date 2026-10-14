// manifest.c — Spacvc manifest binary format reader
#include "manifest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int manifest_dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 2;  // fp16
        case 1: return 4;  // fp32
        case 2: return 4;  // int32
        case 3: return 8;  // int64
        case 4: return 1;  // uint8
        case 5: return 1;  // bool
        default: return 0;
    }
}

static int rd_u32(FILE *f) {
    uint32_t v;
    if (fread(&v, 4, 1, f) != 1) return -1;
    return (int)v;
}

static int rd_i64(FILE *f, int64_t *out) {
    if (fread(out, 8, 1, f) != 1) return -1;
    return 0;
}

static int rd_u64(FILE *f, uint64_t *out) {
    if (fread(out, 8, 1, f) != 1) return -1;
    return 0;
}

int manifest_load(manifest_t *m, const char *spacvc_dir) {
    memset(m, 0, sizeof(*m));
    strncpy(m->base_dir, spacvc_dir, sizeof(m->base_dir) - 1);

    char path[1024];
    snprintf(path, sizeof(path), "%s/manifest.bin", spacvc_dir);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    // Magic
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != MANIFEST_MAGIC) {
        fprintf(stderr, "Bad magic: 0x%08X\n", magic);
        fclose(f);
        return -1;
    }

    // Version
    int version = rd_u32(f);
    if (version != MANIFEST_VERSION) {
        fprintf(stderr, "Bad version: %d\n", version);
        fclose(f);
        return -1;
    }

    // Num tensors
    m->n_tensors = rd_u32(f);
    if (m->n_tensors <= 0 || m->n_tensors > MAX_TENSORS) {
        fprintf(stderr, "Bad tensor count: %d\n", m->n_tensors);
        fclose(f);
        return -1;
    }

    // Config
    m->config.count = rd_u32(f);
    if (m->config.count > 32) m->config.count = 32;
    for (int i = 0; i < m->config.count; i++) {
        rd_i64(f, &m->config.values[i]);
    }

    // Tensors
    for (int i = 0; i < m->n_tensors; i++) {
        manifest_tensor_t *t = &m->tensors[i];
        memset(t, 0, sizeof(*t));

        int name_len = rd_u32(f);
        if (name_len <= 0 || name_len >= MAX_NAME_LEN) {
            fprintf(stderr, "Bad name len: %d at tensor %d\n", name_len, i);
            fclose(f);
            return -1;
        }
        if (fread(t->name, name_len, 1, f) != 1) {
            fclose(f);
            return -1;
        }
        t->name[name_len] = '\0';

        t->ndim = rd_u32(f);
        if (t->ndim < 0 || t->ndim > MAX_NDIM) t->ndim = MAX_NDIM;
        for (int d = 0; d < t->ndim; d++) {
            rd_i64(f, &t->shape[d]);
        }

        t->dtype = rd_u32(f);
        t->data_file = rd_u32(f);
        rd_u64(f, &t->size_bytes);
    }

    fclose(f);
    return 0;
}

int manifest_load_tensor(const manifest_t *m, int idx, void *buf) {
    if (idx < 0 || idx >= m->n_tensors) return -1;
    manifest_tensor_t *t = &m->tensors[idx];

    char path[1024];
    snprintf(path, sizeof(path), "%s/data/%d", m->base_dir, t->data_file);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }

    size_t nread = fread(buf, 1, (size_t)t->size_bytes, f);
    fclose(f);

    if (nread != (size_t)t->size_bytes) {
        fprintf(stderr, "Short read: %zu / %llu\n", nread, (unsigned long long)t->size_bytes);
        return -1;
    }

    return 0;
}

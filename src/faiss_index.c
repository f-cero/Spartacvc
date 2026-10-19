#include "faiss_index.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float l2sq(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

int spcv_index_load(const char* path, SpcvIndex* out) {
    memset(out, 0, sizeof(*out));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[faiss_index] cannot open %s\n", path); return -1; }

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "SPCVIX01", 8) != 0) {
        fprintf(stderr, "[faiss_index] bad magic in %s\n", path); fclose(f); return -1;
    }
    int d, nlist; int64_t ntotal; int metric;
    if (fread(&d, 4, 1, f) != 1 || fread(&nlist, 4, 1, f) != 1 ||
        fread(&ntotal, 8, 1, f) != 1 || fread(&metric, 4, 1, f) != 1) {
        fprintf(stderr, "[faiss_index] short header\n"); fclose(f); return -1;
    }

    size_t nc = (size_t)nlist * d;
    size_t no = (size_t)nlist + 1;
    size_t nb = (size_t)ntotal * d;
    size_t np = (size_t)ntotal;

    float* centroids = (float*)malloc(nc * sizeof(float));
    int64_t* cluster_off = (int64_t*)malloc(no * sizeof(int64_t));
    float* big_npy = (float*)malloc(nb * sizeof(float));
    int64_t* pos2id = (int64_t*)malloc(np * sizeof(int64_t));
    if (!centroids || !cluster_off || !big_npy || !pos2id) {
        fprintf(stderr, "[faiss_index] OOM\n"); fclose(f);
        free(centroids); free(cluster_off); free(big_npy); free(pos2id); return -1;
    }
    if (fread(centroids, sizeof(float), nc, f) != nc ||
        fread(cluster_off, sizeof(int64_t), no, f) != no ||
        fread(big_npy, sizeof(float), nb, f) != nb ||
        fread(pos2id, sizeof(int64_t), np, f) != np) {
        fprintf(stderr, "[faiss_index] short data\n"); fclose(f);
        free(centroids); free(cluster_off); free(big_npy); free(pos2id); return -1;
    }
    fclose(f);

    out->d = d; out->nlist = nlist; out->ntotal = ntotal; out->metric = metric;
    out->centroids = centroids; out->cluster_off = cluster_off;
    out->big_npy = big_npy; out->pos2id = pos2id;
    return 0;
}

void spcv_index_free(SpcvIndex* idx) {
    if (!idx) return;
    free(idx->centroids); free(idx->cluster_off);
    free(idx->big_npy); free(idx->pos2id);
    memset(idx, 0, sizeof(*idx));
}

/* Find the k smallest values in arr[0..n) ; writes their indices to outidx (size k). */
static void topk_smallest(const float* arr, int n, int k, int* outidx) {
    /* n is small (~40); full insertion sort, then take first k */
    int* order = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {
        int key = order[i]; float kv = arr[key];
        int j = i - 1;
        while (j >= 0 && arr[order[j]] > kv) { order[j + 1] = order[j]; j--; }
        order[j + 1] = key;
    }
    for (int i = 0; i < k; i++) outidx[i] = order[i];
    free(order);
}

void spcv_index_retrieve(const SpcvIndex* idx, const float* q, float* out) {
    int d = idx->d;
    /* 1. nearest centroid by L2 */
    int best_c = 0; float best_d = 1e30f;
    for (int c = 0; c < idx->nlist; c++) {
        float dd = l2sq(idx->centroids + (size_t)c * d, q, d);
        if (dd < best_d) { best_d = dd; best_c = c; }
    }
    /* 2. top-8 by L2 in cluster best_c */
    int64_t lo = idx->cluster_off[best_c], hi = idx->cluster_off[best_c + 1];
    int n = (int)(hi - lo);
    if (n <= 0) { for (int i = 0; i < d; i++) out[i] = 0.0f; return; }
    int k = n < 8 ? n : 8;
    float* l2 = (float*)malloc(sizeof(float) * n);
    int* top = (int*)malloc(sizeof(int) * k);
    for (int i = 0; i < n; i++)
        l2[i] = l2sq(idx->big_npy + (lo + i) * d, q, d);
    topk_smallest(l2, n, k, top);
    /* 3. weight = (1/l2)^2 normalized */
    float* w = (float*)malloc(sizeof(float) * k);
    float wsum = 0.0f;
    for (int i = 0; i < k; i++) {
        float s = l2[top[i]]; if (s < 1e-9f) s = 1e-9f;
        w[i] = (1.0f / s) * (1.0f / s); wsum += w[i];
    }
    for (int i = 0; i < k; i++) w[i] /= wsum;
    /* 4. out = sum big_npy[id] * w  (big_npy indexed by ID, as rvc does) */
    for (int i = 0; i < d; i++) out[i] = 0.0f;
    for (int i = 0; i < k; i++) {
        int64_t id = idx->pos2id[lo + top[i]];
        const float* v = idx->big_npy + (size_t)id * d;
        float wi = w[i];
        for (int j = 0; j < d; j++) out[j] += v[j] * wi;
    }
    free(l2); free(top); free(w);
}

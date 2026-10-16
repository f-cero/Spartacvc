/**
 * @file    tensor.c
 * @brief   Minimal float32 tensor ops for RVC v2 inference.
 *
 * All tensors are 4D [N, C, H, W] row-major (last dim contiguous).
 * For 1D sequences we use [1, C, L, 1].
 * For 2D matrices we use [1, C, R, Cols].
 *
 * This is NOT a general tensor library — just the ops the RVC v2
 * voice model needs: Conv1d, ConvTranspose1d, MatMul, LayerNorm,
 * attention, activations, and basic element-wise math.
 */
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Allocation                                                          */
/* ------------------------------------------------------------------ */

t4 *t4_alloc(int ndim, int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
    t4 *t = calloc(1, sizeof(t4));
    t->ndim = ndim;
    t->shape[0] = s0;
    t->shape[1] = s1;
    t->shape[2] = s2;
    t->shape[3] = s3;
    t->total = s0 * s1 * s2 * s3;
    t->data = calloc((size_t)t->total, sizeof(float));
    return t;
}

void t4_free(t4 *t) {
    if (t) { free(t->data); free(t); }
}

void t4_zeros(t4 *t) { memset(t->data, 0, (size_t)t->total * sizeof(float)); }

void t4_fill(t4 *t, float val) {
    for (int64_t i = 0; i < t->total; i++) t->data[i] = val;
}

void t4_copy(t4 *dst, const t4 *src) {
    memcpy(dst->data, src->data, (size_t)src->total * sizeof(float));
    dst->ndim = src->ndim;
    for (int i = 0; i < 4; i++) dst->shape[i] = src->shape[i];
    dst->total = src->total;
}

/* ------------------------------------------------------------------ */
/* Element-wise ops                                                    */
/* ------------------------------------------------------------------ */

void t4_add(t4 *out, const t4 *a, const t4 *b) {
    for (int64_t i = 0; i < a->total; i++) out->data[i] = a->data[i] + b->data[i];
}

void t4_sub(t4 *out, const t4 *a, const t4 *b) {
    for (int64_t i = 0; i < a->total; i++) out->data[i] = a->data[i] - b->data[i];
}

void t4_mul(t4 *out, const t4 *a, const t4 *b) {
    for (int64_t i = 0; i < a->total; i++) out->data[i] = a->data[i] * b->data[i];
}

void t4_mul_scalar(t4 *out, const t4 *a, float s) {
    for (int64_t i = 0; i < a->total; i++) out->data[i] = a->data[i] * s;
}

void t4_div(t4 *out, const t4 *a, const t4 *b) {
    for (int64_t i = 0; i < a->total; i++) out->data[i] = a->data[i] / b->data[i];
}

/* ------------------------------------------------------------------ */
/* Activations                                                         */
/* ------------------------------------------------------------------ */

void t4_relu(t4 *t) {
    for (int64_t i = 0; i < t->total; i++)
        if (t->data[i] < 0) t->data[i] = 0;
}

void t4_leaky_relu(t4 *t, float slope) {
    for (int64_t i = 0; i < t->total; i++)
        if (t->data[i] < 0) t->data[i] *= slope;
}

void t4_sigmoid(t4 *t) {
    for (int64_t i = 0; i < t->total; i++)
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
}

void t4_tanh(t4 *t) {
    for (int64_t i = 0; i < t->total; i++)
        t->data[i] = tanhf(t->data[i]);
}

void t4_softplus(t4 *t) {
    for (int64_t i = 0; i < t->total; i++)
        t->data[i] = log1pf(expf(t->data[i]));
}

/* ------------------------------------------------------------------ */
/* Reduce ops                                                          */
/* ------------------------------------------------------------------ */

/**
 * ReduceL2 along dim=1 (channel): out[N,1,H,W] = sqrt(sum_c(a^2) + eps)
 */
void t4_reduce_l2(t4 *out, const t4 *a, float eps) {
    int64_t N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++) {
                float sum = eps;
                for (int64_t c = 0; c < C; c++) {
                    float v = a->data[((n*C + c)*H + h)*W + w];
                    sum += v * v;
                }
                out->data[(n*H + h)*W + w] = sqrtf(sum);
            }
}

/**
 * ReduceMean along dim=1 (channel): out[N,1,H,W]
 */
void t4_reduce_mean(t4 *out, const t4 *a) {
    int64_t N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++) {
                float sum = 0;
                for (int64_t c = 0; c < C; c++)
                    sum += a->data[((n*C + c)*H + h)*W + w];
                out->data[(n*H + h)*W + w] = sum / (float)C;
            }
}

/* ------------------------------------------------------------------ */
/* LayerNorm                                                           */
/* ------------------------------------------------------------------ */

/**
 * LayerNorm over the last dim (W):
 *   out[n,c,h,w] = (a[n,c,h,w] - mean) / sqrt(var + eps) * gamma[w] + beta[w]
 * where mean/var are computed over w for each (n,c,h).
 *
 * gamma, beta have length W.
 */
void t4_layer_norm(t4 *out, const t4 *a, const float *gamma, const float *beta,
                   int64_t norm_dim, float eps) {
    int64_t N = a->shape[0], C = a->shape[1], H = a->shape[2], W = norm_dim;
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++) {
                float *row = &a->data[((n*C + c)*H + h)*W];
                float mean = 0;
                for (int64_t w = 0; w < W; w++) mean += row[w];
                mean /= (float)W;
                float var = 0;
                for (int64_t w = 0; w < W; w++) {
                    float d = row[w] - mean;
                    var += d * d;
                }
                var /= (float)W;
                float inv = 1.0f / sqrtf(var + eps);
                float *orow = &out->data[((n*C + c)*H + h)*W];
                for (int64_t w = 0; w < W; w++)
                    orow[w] = (row[w] - mean) * inv * gamma[w] + beta[w];
            }
}

/* ------------------------------------------------------------------ */
/* MatMul                                                              */
/* ------------------------------------------------------------------ */

/**
 * MatMul: a[N, R, K] x b[K, C] -> out[N, R, C]
 * (batched 2D matmul, treating shape as [N, R, K] and [K, C])
 *
 * a: shape [1, R, K, 1] stored as [R*K]
 * b: shape [1, K, C, 1] stored as [K*C]
 * out: shape [1, R, C, 1] stored as [R*C]
 */
void t4_matmul(t4 *out, const t4 *a, const t4 *b) {
    /* a: [N, R, K] flattened to N*R*K elements */
    /* b: [K, C] flattened to K*C elements */
    int64_t N = a->shape[0];
    int64_t R = a->shape[1];
    int64_t K = a->shape[2];
    int64_t C = b->shape[1];

    for (int64_t n = 0; n < N; n++)
        for (int64_t r = 0; r < R; r++)
            for (int64_t c = 0; c < C; c++) {
                float sum = 0;
                for (int64_t k = 0; k < K; k++)
                    sum += a->data[(n*R + r)*K + k] * b->data[k*C + c];
                out->data[(n*R + r)*C + c] = sum;
            }
}

/* ------------------------------------------------------------------ */
/* Conv1d                                                              */
/* ------------------------------------------------------------------ */

/**
 * Conv1d: input[N, C_in, L], weight[C_out, C_in, K], bias[C_out]
 * padding = K/2, stride=1, groups=1
 * out[N, C_out, L]
 *
 * input:  shape [1, C_in, L, 1]
 * weight: shape [1, C_out, C_in*K, 1] (PyTorch layout: [C_out, C_in, K])
 * bias:   [C_out]
 */
void t4_conv1d(t4 *out, const t4 *input, const t4 *weight, const float *bias,
               int64_t kernel_size, int64_t padding) {
    int64_t N = 1;
    int64_t C_in  = input->shape[1];
    int64_t L     = input->shape[2];
    int64_t C_out = weight->shape[1];

    for (int64_t co = 0; co < C_out; co++) {
        float b = bias ? bias[co] : 0;
        for (int64_t l = 0; l < L; l++) {
            float sum = b;
            for (int64_t ci = 0; ci < C_in; ci++) {
                const float *wptr = &weight->data[(co*C_in + ci)*kernel_size];
                for (int64_t k = 0; k < kernel_size; k++) {
                    int64_t pos = l - padding + k;
                    if (pos >= 0 && pos < L)
                        sum += input->data[(ci*L + pos)] * wptr[k];
                }
            }
            out->data[(co*L + l)] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/* ConvTranspose1d (transpose convolution / upsampling)                */
/* ------------------------------------------------------------------ */

/**
 * ConvTranspose1d: input[N, C_in, L], weight[C_in, C_out, K]
 * out[N, C_out, L*K]
 *
 * This is the "unfolding" operation: each input sample is placed
 * at position l*K and multiplied by the full kernel.
 *
 * input:  shape [1, C_in, L, 1]
 * weight: shape [1, C_in, C_out*K, 1] (PyTorch: [C_in, C_out, K])
 */
void t4_conv_transpose1d(t4 *out, const t4 *input, const t4 *weight,
                         const float *bias, int64_t kernel_size) {
    int64_t C_in  = input->shape[1];
    int64_t L     = input->shape[2];
    int64_t C_out = weight->shape[2] / kernel_size; /* weight is [C_in, C_out*K] */

    /* Zero output */
    t4_zeros(out);

    for (int64_t ci = 0; ci < C_in; ci++)
        for (int64_t l = 0; l < L; l++) {
            float val = input->data[ci*L + l];
            if (val == 0) continue;
            const float *wptr = &weight->data[ci * (C_out * kernel_size)];
            for (int64_t co = 0; co < C_out; co++)
                for (int64_t k = 0; k < kernel_size; k++)
                    out->data[(co * (L * kernel_size)) + l * kernel_size + k]
                        += val * wptr[co * kernel_size + k];
        }

    if (bias)
        for (int64_t co = 0; co < C_out; co++)
            for (int64_t l = 0; l < L * kernel_size; l++)
                out->data[co * (L * kernel_size) + l] += bias[co];
}

/* ------------------------------------------------------------------ */
/* Gather                                                              */
/* ------------------------------------------------------------------ */

/**
 * Gather along dim=1: out[i] = a[indices[i]]
 * a: [1, C, 1, 1] (embedding table), indices: [N]
 * out: [1, N, 1, 1]
 */
void t4_gather(t4 *out, const t4 *a, const int64_t *indices, int64_t n) {
    int64_t C = a->shape[1]; /* embedding dim */
    for (int64_t i = 0; i < n; i++) {
        int64_t idx = indices[i];
        memcpy(&out->data[i * C], &a->data[idx * C], C * sizeof(float));
    }
}

/* ------------------------------------------------------------------ */
/* Reshape                                                             */
/* ------------------------------------------------------------------ */

void t4_reshape(t4 *t, int ndim, int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
    t->ndim = ndim;
    t->shape[0] = s0;
    t->shape[1] = s1;
    t->shape[2] = s2;
    t->shape[3] = s3;
    t->total = s0 * s1 * s2 * s3;
}

/* ------------------------------------------------------------------ */
/* Transpose dims 1 and 2: [N,C,H,W] -> [N,H,C,W]                     */
/* ------------------------------------------------------------------ */

void t4_transpose_12(t4 *out, const t4 *a) {
    int64_t N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++)
                    out->data[((n*H + h)*C + c)*W + w] = a->data[((n*C + c)*H + h)*W + w];
}

/* ------------------------------------------------------------------ */
/* Softmax along last dim                                              */
/* ------------------------------------------------------------------ */

void t4_softmax(t4 *t) {
    /* Assume shape [N, C, H, W] — softmax over W */
    int64_t N = t->shape[0], C = t->shape[1], H = t->shape[2], W = t->shape[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++) {
                float *row = &t->data[((n*C + c)*H + h)*W];
                float maxv = row[0];
                for (int64_t w = 1; w < W; w++) if (row[w] > maxv) maxv = row[w];
                float sum = 0;
                for (int64_t w = 0; w < W; w++) { row[w] = expf(row[w] - maxv); sum += row[w]; }
                for (int64_t w = 0; w < W; w++) row[w] /= sum;
            }
}

/* ------------------------------------------------------------------ */
/* Slice along dim=2 (H): out = a[:, :, start:end, :]                 */
/* ------------------------------------------------------------------ */

void t4_slice_h(t4 *out, const t4 *a, int64_t start, int64_t end) {
    int64_t N = a->shape[0], C = a->shape[1], W = a->shape[3];
    int64_t len = end - start;
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < len; h++)
                memcpy(&out->data[((n*C + c)*len + h)*W],
                       &a->data[((n*C + c)*(a->shape[2]) + start + h)*W],
                       W * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* Pad along dim=2 (H)                                                */
/* ------------------------------------------------------------------ */

void t4_pad_h(t4 *out, const t4 *a, int64_t pre, int64_t post) {
    int64_t N = a->shape[0], C = a->shape[1], H = a->shape[2], W = a->shape[3];
    t4_zeros(out);
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                memcpy(&out->data[((n*C + c)*(H + pre + post) + pre + h)*W],
                       &a->data[((n*C + c)*H + h)*W],
                       W * sizeof(float));
}

/* ------------------------------------------------------------------ */
/* Random normal (Box-Muller)                                          */
/* ------------------------------------------------------------------ */

void t4_random_normal(t4 *t, uint32_t *seed) {
    for (int64_t i = 0; i < t->total; i += 2) {
        /* Simple LCG for reproducibility */
        *seed = *seed * 1664525u + 1013904223u;
        float u1 = (float)(*seed & 0xFFFFFF) / (float)0xFFFFFF + 0.0f;
        *seed = *seed * 1664525u + 1013904223u;
        float u2 = (float)(*seed & 0xFFFFFF) / (float)0xFFFFFF + 0.0f;
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * (float)M_PI * u2;
        t->data[i] = r * cosf(theta);
        if (i + 1 < t->total) t->data[i+1] = r * sinf(theta);
    }
}

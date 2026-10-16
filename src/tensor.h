// tensor.h — minimal float32 tensor ops for RVC inference
#ifndef SPARTACVC_TENSOR_H
#define SPARTACVC_TENSOR_H

#include <stddef.h>
#include <stdint.h>

/* 4D tensor: [N, C, H, W] row-major (last dim contiguous) */
typedef struct {
    float  *data;
    int     ndim;
    int64_t shape[4];
    int64_t total;  /* product of shape */
} t4;

/* Allocate a 4D tensor. shape[4] must be set, ndim set. */
t4 *t4_alloc(int ndim, int64_t s0, int64_t s1, int64_t s2, int64_t s3);
void t4_free(t4 *t);
void t4_zeros(t4 *t);

/* Element-wise ops (same shape) */
void t4_add(t4 *out, const t4 *a, const t4 *b);
void t4_sub(t4 *out, const t4 *a, const t4 *b);
void t4_mul(t4 *out, const t4 *a, const t4 *b);
void t4_mul_scalar(t4 *out, const t4 *a, float s);
void t4_div(t4 *out, const t4 *a, const t4 *b);

/* Activation functions (in-place) */
void t4_relu(t4 *t);
void t4_leaky_relu(t4 *t, float slope);
void t4_sigmoid(t4 *t);
void t4_tanh(t4 *t);
void t4_softplus(t4 *t);

/* ReduceL2 along dim=1 (channel dim): out[N,1,H,W] = sqrt(sum_c(a^2)+eps) */
void t4_reduce_l2(t4 *out, const t4 *a, float eps);

/* ReduceMean along dim=1: out[N,1,H,W] */
void t4_reduce_mean(t4 *out, const t4 *a);

/* LayerNorm over last dim (dim=3): out = (x-mean)/sqrt(var+eps) * gamma + beta
 * a: [N,C,H,W], gamma/beta: [W] */
void t4_layer_norm(t4 *out, const t4 *a, const float *gamma, const float *beta,
                   int64_t norm_dim, float eps);

/* MatMul: a[N,C1,K] x b[K,C2] -> out[N,C2]  (2D matmul) */
void t4_matmul(t4 *out, const t4 *a, const t4 *b);

/* Conv1d: input[N,C_in,L], weight[C_out,C_in,K], bias[C_out]
 * padding = K/2, stride=1, groups=1
 * out[N,C_out,L] */
void t4_conv1d(t4 *out, const t4 *input, const t4 *weight, const float *bias,
               int64_t kernel_size, int64_t padding);

/* ConvTranspose1d (upsample): input[N,C_in,L], weight[C_in,C_out,K]
 * out[N,C_out,L*K] */
void t4_conv_transpose1d(t4 *out, const t4 *input, const t4 *weight,
                         const float *bias, int64_t kernel_size);

/* Gather along dim=0: out[i] = a[indices[i]]  (1D indices) */
void t4_gather(t4 *out, const t4 *a, const int64_t *indices, int64_t n);

/* Reshape (data stays same, just change shape interpretation) */
void t4_reshape(t4 *t, int ndim, int64_t s0, int64_t s1, int64_t s2, int64_t s3);

/* Transpose dims 1 and 2: [N,C,H,W] -> [N,H,C,W] */
void t4_transpose_12(t4 *out, const t4 *a);

/* Softmax along last dim */
void t4_softmax(t4 *t);

/* Slice along dim=2 (H): out = a[:, :, start:end, :] */
void t4_slice_h(t4 *out, const t4 *a, int64_t start, int64_t end);

/* Pad along dim=2 (H): prepend 'pre' zeros, append 'post' zeros */
void t4_pad_h(t4 *out, const t4 *a, int64_t pre, int64_t post);

/* Copy */
void t4_copy(t4 *dst, const t4 *src);

/* Fill with value */
void t4_fill(t4 *t, float val);

/* Random normal (Box-Muller, seed-based) */
void t4_random_normal(t4 *t, uint32_t *seed);

#endif

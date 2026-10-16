/**
 * @file    rvc.h
 * @brief   RVC v2 voice model — forward pass in pure C.
 *
 * Architecture: SynthesizerTrnMsNSFsid (RVC v2)
 *
 *   phone [1,T,768] + pitch [1,T]
 *     → enc_p (encoder with pitch conditioning)
 *     → flow  (4× conditional flow blocks)
 *     → dec   (decoder: upsampling + ResBlocks + noise → waveform)
 *
 * All weights are loaded from a .pth file via pth_loader.
 * No ONNX Runtime needed for the voice model.
 */
#ifndef SPARTACVC_RVC_H
#define SPARTACVC_RVC_H

#include <stdint.h>
#include "tensor.h"
#include "pth_loader.h"

/* ------------------------------------------------------------------ */
/* Model config (from .pth config array)                               */
/* ------------------------------------------------------------------ */

typedef struct {
    int n_mel;            /* mel bins (1025 for 768-dim)            */
    int filter_channels;  /* flow filter channels (32)              */
    int ff_dim;           /* feedforward hidden dim (192)           */
    int p_dim;            /* encoder dim (192)                      */
    int ques;             /* embedding dim (768 or 256)             */
    int p_kernel;         /* encoder kernel size                    */
    int p_heads;          /* attention heads (6)                    */
    int p_layers;         /* encoder layers (3)                     */
    int upsample_rates[8];   /* e.g. [12,10,2,2]                   */
    int n_upsample;         /* number of upsample stages (4)        */
    int upsample_initial;   /* initial channels (512)               */
    int upsample_kernels[8]; /* e.g. [24,20,4,4]                   */
    int resblock_kernel;    /* resblock kernel (3,7,11)             */
    int resblock_channels;  /* resblock channels (256)              */
    int sr;               /* output sample rate (48000)            */
    int version;          /* 2 = v2                                 */
} rvc_config_t;

/* ------------------------------------------------------------------ */
/* Model — holds all weights + config                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    rvc_config_t cfg;
    pth_model_t  pth;       /* .pth file handle                     */

    /* --- enc_p weights (pointers into pth tensor data) --- */
    float *emb_phone_w;     /* [p_dim, ques]                        */
    float *emb_phone_b;     /* [p_dim]                              */
    float *emb_pitch_w;     /* [n_mel+1, p_dim]                     */
    int    n_attn_layers;
    /* Per attention layer: q,k,v,o conv weights + biases */
    float *attn_qw[16];     /* [p_dim, p_dim, 1]                    */
    float *attn_qb[16];     /* [p_dim]                              */
    float *attn_kw[16];
    float *attn_kb[16];
    float *attn_vw[16];
    float *attn_vb[16];
    float *attn_ow[16];
    float *attn_ob[16];
    /* Per FFN layer: conv1, conv2 weights + biases */
    float *ffn1w[16];       /* [ff_dim, p_dim, 3]                   */
    float *ffn1b[16];       /* [ff_dim]                             */
    float *ffn2w[16];       /* [p_dim, ff_dim, 3]                   */
    float *ffn2b[16];       /* [p_dim]                              */
    /* LayerNorm params */
    float *ln1_gamma[16];   /* [p_dim]                              */
    float *ln1_beta[16];
    float *ln2_gamma[16];
    float *ln2_beta[16];
    /* Relative position embeddings (per attention layer) */
    float *attn_relk[16];   /* [max_rel, head_dim]                 */
    float *attn_relv[16];   /* [max_rel, head_dim]                 */
    int    attn_max_rel;    /* 21 (window_size*2+1)                */
    int    attn_head_dim;   /* p_dim / p_heads                     */
    /* Projection */
    float *proj_w;          /* [2*p_dim, p_dim, 1]                  */
    float *proj_b;          /* [2*p_dim]                            */

    /* --- flow weights --- */
    int    n_flow_blocks;
    /* Per flow block: in_layers (3), res_skip (3), cond, pre, post */
    float *flow_in_w[16][4];    /* [2*filter_ch, p_dim, 5]          */
    float *flow_in_b[16][4];    /* [2*filter_ch]                    */
    float *flow_in_g[16][4];    /* [2*filter_ch,1,1]                */
    float *flow_rs_w[16][4];    /* [filter_ch, filter_ch, 1]        */
    float *flow_rs_b[16][4];
    float *flow_rs_g[16][4];
    float *flow_cond_w[16];     /* [12*filter_ch, 1, 1]             */
    float *flow_cond_v[16];     /* [12*filter_ch, p_dim, 1]         */
    float *flow_cond_g[16];     /* [12*filter_ch, 1, 1] weight_norm */
    float *flow_cond_b[16];     /* [12*filter_ch]                   */
    float *flow_pre_w[16];      /* [p_dim, filter_ch, 1]            */
    float *flow_pre_b[16];
    float *flow_post_w[16];     /* [filter_ch/2, p_dim, 1]          */
    float *flow_post_b[16];

    /* --- dec weights --- */
    float *dec_conv_pre_w;      /* [upsample_initial, p_dim, 7]     */
    float *dec_conv_pre_b;      /* [upsample_initial]               */
    float *dec_cond_w;          /* [upsample_initial, p_dim, 1]     */
    float *dec_cond_b;          /* [upsample_initial]               */
    /* Upsampling layers */
    float *ups_w[8];            /* [ch_in, ch_out, kernel]          */
    float *ups_g[8];            /* [ch_in, 1, 1]                    */
    float *ups_b[8];            /* [ch_out]                         */
    /* ResBlocks (12 total) */
    float *rb_conv1_w[12][3];   /* [ch, ch, kernel]                 */
    float *rb_conv1_b[12][3];
    float *rb_conv1_g[12][3];
    float *rb_conv2_w[12][3];   /* [ch, ch, kernel]                 */
    float *rb_conv2_b[12][3];
    float *rb_conv2_g[12][3];
    /* Noise convolutions */
    float *noise_w[4];          /* [ch, 1, kernel]                  */
    float *noise_b[4];
    /* Speaker embedding */
    float *emb_g;               /* [256] speaker embedding          */
    int    emb_g_dim;           /* 256                              */
    /* m_source */
    float *m_source_w;          /* [1,1]                            */
    float *m_source_b;          /* [1]                              */
    /* conv_post */
    float *conv_post_w;         /* [1, 32, 7]                       */
    /* Channel sizes per resblock group */
    int    rb_channels[12];
    int    rb_kernels[12];

    /* Derived */
    int    fps;                /* sr / prod(upsample_rates)         */
    int    total_channels;     /* upsample_initial                  */
} rvc_model_t;

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief  Load an RVC v2 model from a .pth file.
 *
 * Reads the config array, resolves all tensor names, and stores
 * pointers into the .pth data. Caller must rvc_free() when done.
 *
 * @param  m     model struct (zeroed before call)
 * @param  path  path to .pth file
 * @return 0 on success, -1 on error
 */
int rvc_load(rvc_model_t *m, const char *path);

/**
 * @brief  Free all model resources.
 */
void rvc_free(rvc_model_t *m);

/**
 * @brief  Run the full RVC v2 forward pass.
 *
 * @param  m          loaded model
 * @param  phone      HuBERT embeddings [T, ques] (row-major)
 * @param  T          number of time frames
 * @param  pitch      quantized pitch [T] (int, 1..255)
 * @param  pitchf     raw f0 in Hz [T]
 * @param  noise      random noise [192, T] (row-major)
 * @param  out        output buffer (caller allocates: T * prod(upsample_rates) floats)
 * @param  out_len    [out] number of samples written
 * @param  seed       RNG seed for noise (mutable)
 * @return 0 on success, -1 on error
 */
int rvc_infer(const rvc_model_t *m,
              const float *phone, int T,
              const int64_t *pitch, const float *pitchf,
              const float *noise,
              float *out, int *out_len,
              uint32_t *seed);

#endif

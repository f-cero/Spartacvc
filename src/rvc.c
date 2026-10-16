/**
 * @file    rvc.c
 * @brief   RVC v2 voice model — load weights from .pth.
 * Forward pass is in rvc_infer.c.
 */
#include "rvc.h"
#include "debug.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int load_count = 0;
static float *load_w(rvc_model_t *m, const char *name, int64_t *out_numel) {
    const pth_tensor_t *t = pth_find_tensor(&m->pth, name);
    if (!t) { fprintf(stderr, "[rvc] MISSING: %s\n", name); return NULL; }
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    float *buf = malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;
    if (pth_read_tensor_f32(&m->pth, t, buf) != 0) { fprintf(stderr, "[rvc] READ FAIL: %s\n", name); free(buf); return NULL; }
    load_count++;
    if (load_count % 50 == 0) fprintf(stderr, "[rvc] loaded %d tensors\n", load_count);
    if (out_numel) *out_numel = n;
    return buf;
}

static float *lw(rvc_model_t *m, const char *fmt, int idx) {
    char nm[256];
    snprintf(nm, sizeof(nm), fmt, idx);
    return load_w(m, nm, NULL);
}

int rvc_load(rvc_model_t *m, const char *path) {
    memset(m, 0, sizeof(*m));
    if (pth_load(&m->pth, path) != 0) {
        fprintf(stderr, "[rvc] open fail: %s\n", path);
        return -1;
    }

    /* Debug: print first 5 tensor names */
    for (int i = 0; i < 5 && i < m->pth.pkl.n_tensors; i++) {
        fprintf(stderr, "  [dbg] tensor[%d] name='%s' ndim=%d\n", i, m->pth.pkl.tensors[i].name, m->pth.pkl.tensors[i].ndim);
    }

    /* Use config from pickle parser if available */
    if (m->pth.pkl.has_config) {
        pth_config_t *pc = &m->pth.pkl.config;
        m->cfg.n_mel = pc->n_mel;
        m->cfg.filter_channels = pc->filter_channels;
        m->cfg.ff_dim = pc->ff_dim;
        m->cfg.p_dim = pc->p_dim;
        m->cfg.ques = pc->ques;
        m->cfg.p_kernel = pc->p_kernel;
        m->cfg.p_heads = pc->p_heads;
        m->cfg.p_layers = pc->p_layers;
        m->cfg.n_upsample = pc->n_upsample;
        m->cfg.upsample_initial = pc->upsample_initial;
        m->cfg.sr = pc->sr;
        for (int i = 0; i < pc->n_upsample; i++) {
            m->cfg.upsample_rates[i] = pc->upsample_rates[i];
            m->cfg.upsample_kernels[i] = pc->upsample_kernels[i];
        }
    }

    fprintf(stderr, "[rvc] config loaded: n_mel=%d p_dim=%d ques=%d sr=%d\n",
            m->cfg.n_mel, m->cfg.p_dim, m->cfg.ques, m->cfg.sr);

    /* Fallbacks from weight shapes */
    fprintf(stderr, "[rvc] fallbacks...\n");
    if (m->cfg.p_dim == 0) {
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "enc_p.emb_phone.bias");
        fprintf(stderr, "[rvc] emb_phone.bias: %p\n", (void*)t);
        if (t) m->cfg.p_dim = (int)t->shape[0];
    }
    fprintf(stderr, "[rvc] p_dim=%d\n", m->cfg.p_dim);
    if (m->cfg.ques == 0) {
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "enc_p.emb_phone.weight");
        if (t && t->ndim == 2) m->cfg.ques = (int)t->shape[1];
    }
    if (m->cfg.n_mel == 0) {
        /* n_mel is the mel dimension, NOT the pitch bins.
         * emb_pitch.weight shape is [n_pitch_bins, p_dim] where n_pitch_bins = n_mel + 1
         * But we can also get it from the flow pre layer: [p_dim, n_mel, 1]
         */
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "flow.flows.0.pre.weight");
        if (t && t->ndim == 3) m->cfg.n_mel = (int)t->shape[1];
        else {
            t = pth_find_tensor(&m->pth, "enc_p.emb_pitch.weight");
            if (t && t->ndim == 2) m->cfg.n_mel = (int)t->shape[0] - 1;
        }
    }
    if (m->cfg.p_layers == 0) {
        int cnt = 0;
        for (int i = 0; i < 16; i++) {
            char nm[256];
            snprintf(nm, sizeof(nm), "enc_p.encoder.attn_layers.%d.conv_q.weight", i);
            if (pth_find_tensor(&m->pth, nm)) cnt = i + 1;
            else break;
        }
        m->cfg.p_layers = cnt ? cnt : 3;
    }
    if (m->cfg.n_upsample == 0) {
        int cnt = 0;
        for (int i = 0; i < 8; i++) {
            char nm[256];
            snprintf(nm, sizeof(nm), "dec.ups.%d.weight_v", i);
            if (pth_find_tensor(&m->pth, nm)) cnt = i + 1;
            else break;
        }
        m->cfg.n_upsample = cnt;
    }
    fprintf(stderr, "[rvc] n_upsample=%d\n", m->cfg.n_upsample);
    if (m->cfg.upsample_initial == 0) {
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "dec.conv_pre.weight");
        if (t && t->ndim == 3) m->cfg.upsample_initial = (int)t->shape[0];
    }
    if (m->cfg.sr == 0) m->cfg.sr = 48000;
    
    /* Derive filter_channels and ff_dim from tensor shapes if not set */
    if (m->cfg.filter_channels == 0) {
        /* FC is the input channels of in_layers, which equals p_dim in RVC v2 */
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "flow.flows.0.enc.in_layers.0.weight_v");
        if (t && t->ndim == 3) m->cfg.filter_channels = (int)t->shape[1];
    }
    if (m->cfg.ff_dim == 0) {
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "enc_p.encoder.ffn_layers.0.conv_1.weight");
        if (t && t->ndim == 3) m->cfg.ff_dim = (int)t->shape[0];
    }
    if (m->cfg.p_heads == 0) {
        /* p_heads = p_dim / head_dim, typically p_dim/4 or p_dim/6 */
        /* From emb_rel_k shape: [1, max_rel, head_dim] where head_dim = p_dim / p_heads */
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "enc_p.encoder.attn_layers.0.emb_rel_k");
        if (t && t->ndim == 3 && m->cfg.p_dim > 0) {
            int head_dim = (int)t->shape[2];
            m->cfg.p_heads = m->cfg.p_dim / head_dim;
        }
        if (m->cfg.p_heads == 0) m->cfg.p_heads = 2;
    }
    if (m->cfg.p_kernel == 0) {
        const pth_tensor_t *t = pth_find_tensor(&m->pth, "enc_p.encoder.attn_layers.0.conv_q.weight");
        if (t && t->ndim == 3) m->cfg.p_kernel = (int)t->shape[2];
    }
    
    /* Derive upsample rates from tensor shapes if not set */
    if (m->cfg.upsample_rates[0] == 0) {
        for (int i = 0; i < m->cfg.n_upsample; i++) {
            char nm[256];
            snprintf(nm, sizeof(nm), "dec.ups.%d.weight_v", i);
            const pth_tensor_t *t = pth_find_tensor(&m->pth, nm);
            if (t && t->ndim == 3) {
                /* weight_v shape: [out_ch, in_ch, kernel] where kernel = 2*rate-1 or rate */
                /* Actually for RVC: upsample rate = kernel_size / 2 (for even kernels) */
                /* The kernel size is shape[2], and rate = kernel / 2 */
                int k = (int)t->shape[2];
                m->cfg.upsample_rates[i] = k / 2;
                m->cfg.upsample_kernels[i] = k;
            }
        }
    }
    fprintf(stderr, "[rvc] upsample_initial=%d sr=%d n_upsample=%d rates=", 
            m->cfg.upsample_initial, m->cfg.sr, m->cfg.n_upsample);
    for (int i = 0; i < m->cfg.n_upsample; i++) fprintf(stderr, "%d ", m->cfg.upsample_rates[i]);
    fprintf(stderr, "\n");

    /* Compute fps */
    int prod = 1;
    for (int i = 0; i < m->cfg.n_upsample; i++) prod *= m->cfg.upsample_rates[i];
    if (prod == 0) prod = 1; // safety
    m->fps = m->cfg.sr / prod;
    m->cfg.version = 2;
    fprintf(stderr, "[rvc] fps=%d\n", m->fps);

    SPCV_TRACE("rvc", "cfg: n_mel=%d fc=%d ff=%d p=%d q=%d L=%d H=%d ups=%d sr=%d fps=%d",
             m->cfg.n_mel, m->cfg.filter_channels, m->cfg.ff_dim,
             m->cfg.p_dim, m->cfg.ques, m->cfg.p_layers, m->cfg.p_heads,
             m->cfg.n_upsample, m->cfg.sr, m->fps);

    /* --- enc_p weights --- */
    fprintf(stderr, "[rvc] loading enc_p weights...\n");
    m->emb_phone_w = load_w(m, "enc_p.emb_phone.weight", NULL);
    fprintf(stderr, "[rvc] emb_phone_w=%p\n", (void*)m->emb_phone_w);
    m->emb_phone_b = load_w(m, "enc_p.emb_phone.bias", NULL);
    m->emb_pitch_w = load_w(m, "enc_p.emb_pitch.weight", NULL);
    if (!m->emb_phone_w || !m->emb_phone_b || !m->emb_pitch_w) { rvc_free(m); return -1; }

    m->n_attn_layers = m->cfg.p_layers;
    for (int i = 0; i < m->n_attn_layers; i++) {
        m->attn_qw[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_q.weight", i);
        m->attn_qb[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_q.bias", i);
        m->attn_kw[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_k.weight", i);
        m->attn_kb[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_k.bias", i);
        m->attn_vw[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_v.weight", i);
        m->attn_vb[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_v.bias", i);
        m->attn_ow[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_o.weight", i);
        m->attn_ob[i] = lw(m, "enc_p.encoder.attn_layers.%d.conv_o.bias", i);
        m->ffn1w[i] = lw(m, "enc_p.encoder.ffn_layers.%d.conv_1.weight", i);
        m->ffn1b[i] = lw(m, "enc_p.encoder.ffn_layers.%d.conv_1.bias", i);
        m->ffn2w[i] = lw(m, "enc_p.encoder.ffn_layers.%d.conv_2.weight", i);
        m->ffn2b[i] = lw(m, "enc_p.encoder.ffn_layers.%d.conv_2.bias", i);
        m->ln1_gamma[i] = lw(m, "enc_p.encoder.norm_layers_1.%d.gamma", i);
        m->ln1_beta[i] = lw(m, "enc_p.encoder.norm_layers_1.%d.beta", i);
        m->ln2_gamma[i] = lw(m, "enc_p.encoder.norm_layers_2.%d.gamma", i);
        m->ln2_beta[i] = lw(m, "enc_p.encoder.norm_layers_2.%d.beta", i);
        m->attn_relk[i] = lw(m, "enc_p.encoder.attn_layers.%d.emb_rel_k", i);
        m->attn_relv[i] = lw(m, "enc_p.encoder.attn_layers.%d.emb_rel_v", i);
    }
    m->attn_max_rel = 21;
    m->attn_head_dim = m->cfg.p_dim / m->cfg.p_heads;
    m->proj_w = load_w(m, "enc_p.proj.weight", NULL);
    m->proj_b = load_w(m, "enc_p.proj.bias", NULL);
    if (!m->proj_w || !m->proj_b) { rvc_free(m); return -1; }

    /* --- flow weights (blocks at indices 0,2,4,6) --- */
    int fidx[] = {0, 2, 4, 6};
    m->n_flow_blocks = 0;
    for (int fi = 0; fi < 4; fi++) {
        int idx = fidx[fi];
        char nm[256];
        snprintf(nm, sizeof(nm), "flow.flows.%d.pre.weight", idx);
        if (!pth_find_tensor(&m->pth, nm)) continue;
        int s = m->n_flow_blocks++;
        for (int j = 0; j < 3; j++) {
            char nm2[256];
            snprintf(nm2, sizeof(nm2), "flow.flows.%d.enc.in_layers.%d.weight_v", idx, j);
            m->flow_in_w[s][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "flow.flows.%d.enc.in_layers.%d.bias", idx, j);
            m->flow_in_b[s][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "flow.flows.%d.enc.in_layers.%d.weight_g", idx, j);
            m->flow_in_g[s][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "flow.flows.%d.enc.res_skip_layers.%d.weight_v", idx, j);
            m->flow_rs_w[s][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "flow.flows.%d.enc.res_skip_layers.%d.bias", idx, j);
            m->flow_rs_b[s][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "flow.flows.%d.enc.res_skip_layers.%d.weight_g", idx, j);
            m->flow_rs_g[s][j] = load_w(m, nm2, NULL);
        }
        snprintf(nm, sizeof(nm), "flow.flows.%d.enc.cond_layer.weight_v", idx);
        m->flow_cond_v[s] = load_w(m, nm, NULL);
        snprintf(nm, sizeof(nm), "flow.flows.%d.enc.cond_layer.weight_g", idx);
        m->flow_cond_g[s] = load_w(m, nm, NULL);
        snprintf(nm, sizeof(nm), "flow.flows.%d.enc.cond_layer.bias", idx);
        m->flow_cond_b[s] = load_w(m, nm, NULL);
        snprintf(nm, sizeof(nm), "flow.flows.%d.pre.weight", idx);
        m->flow_pre_w[s] = load_w(m, nm, NULL);
        snprintf(nm, sizeof(nm), "flow.flows.%d.pre.bias", idx);
        m->flow_pre_b[s] = load_w(m, nm, NULL);
        snprintf(nm, sizeof(nm), "flow.flows.%d.post.weight", idx);
        m->flow_post_w[s] = load_w(m, nm, NULL);
        snprintf(nm, sizeof(nm), "flow.flows.%d.post.bias", idx);
        m->flow_post_b[s] = load_w(m, nm, NULL);
    }

    /* --- dec weights --- */
    m->dec_conv_pre_w = load_w(m, "dec.conv_pre.weight", NULL);
    m->dec_conv_pre_b = load_w(m, "dec.conv_pre.bias", NULL);
    m->dec_cond_w = load_w(m, "dec.cond.weight", NULL);
    m->dec_cond_b = load_w(m, "dec.cond.bias", NULL);

    for (int i = 0; i < m->cfg.n_upsample; i++) {
        m->ups_w[i] = lw(m, "dec.ups.%d.weight_v", i);
        m->ups_g[i] = lw(m, "dec.ups.%d.weight_g", i);
        m->ups_b[i] = lw(m, "dec.ups.%d.bias", i);
    }

    for (int i = 0; i < 12; i++) {
        char nm[256];
        snprintf(nm, sizeof(nm), "dec.resblocks.%d.convs1.0.weight_v", i);
        const pth_tensor_t *rw = pth_find_tensor(&m->pth, nm);
        if (!rw) continue;
        m->rb_channels[i] = (int)rw->shape[0];
        m->rb_kernels[i] = (int)rw->shape[2];
        for (int j = 0; j < 3; j++) {
            char nm2[256];
            snprintf(nm2, sizeof(nm2), "dec.resblocks.%d.convs1.%d.weight_v", i, j);
            m->rb_conv1_w[i][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "dec.resblocks.%d.convs1.%d.bias", i, j);
            m->rb_conv1_b[i][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "dec.resblocks.%d.convs1.%d.weight_g", i, j);
            m->rb_conv1_g[i][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "dec.resblocks.%d.convs2.%d.weight_v", i, j);
            m->rb_conv2_w[i][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "dec.resblocks.%d.convs2.%d.bias", i, j);
            m->rb_conv2_b[i][j] = load_w(m, nm2, NULL);
            snprintf(nm2, sizeof(nm2), "dec.resblocks.%d.convs2.%d.weight_g", i, j);
            m->rb_conv2_g[i][j] = load_w(m, nm2, NULL);
        }
    }

    for (int i = 0; i < 4; i++) {
        m->noise_w[i] = lw(m, "dec.noise_convs.%d.weight", i);
        m->noise_b[i] = lw(m, "dec.noise_convs.%d.bias", i);
    }
    /* Speaker embedding (row 0 of emb_g.weight [109, 256]) */
    {
        const pth_tensor_t *eg = pth_find_tensor(&m->pth, "emb_g.weight");
        if (eg && eg->ndim == 2) {
            m->emb_g_dim = (int)eg->shape[1];
            int64_t total = eg->shape[0] * eg->shape[1];
            float *full = malloc((size_t)total * sizeof(float));
            if (full && pth_read_tensor_f32(&m->pth, eg, full) == 0) {
                m->emb_g = malloc((size_t)m->emb_g_dim * sizeof(float));
                memcpy(m->emb_g, full, (size_t)m->emb_g_dim * sizeof(float));
            }
            free(full);
        }
    }
    m->m_source_w = load_w(m, "dec.m_source.l_linear.weight", NULL);
    m->m_source_b = load_w(m, "dec.m_source.l_linear.bias", NULL);
    m->conv_post_w = load_w(m, "dec.conv_post.weight", NULL);

    SPCV_TRACE("rvc", "rvc_load OK: %d flow, %d ups, rb=[%d,%d,%d,%d,%d,%d]",
             m->n_flow_blocks, m->cfg.n_upsample,
             m->rb_channels[0], m->rb_channels[3], m->rb_channels[6],
             m->rb_channels[9], m->rb_channels[10], m->rb_channels[11]);
    return 0;
}

void rvc_free(rvc_model_t *m) {
    if (!m) return;
    free(m->emb_phone_w); free(m->emb_phone_b); free(m->emb_pitch_w);
    for (int i = 0; i < 16; i++) {
        free(m->attn_qw[i]); free(m->attn_qb[i]);
        free(m->attn_kw[i]); free(m->attn_kb[i]);
        free(m->attn_vw[i]); free(m->attn_vb[i]);
        free(m->attn_ow[i]); free(m->attn_ob[i]);
        free(m->ffn1w[i]); free(m->ffn1b[i]);
        free(m->ffn2w[i]); free(m->ffn2b[i]);
        free(m->ln1_gamma[i]); free(m->ln1_beta[i]);
        free(m->ln2_gamma[i]); free(m->ln2_beta[i]);
        free(m->attn_relk[i]); free(m->attn_relv[i]);
    }
    free(m->proj_w); free(m->proj_b);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 4; j++) {
            free(m->flow_in_w[i][j]); free(m->flow_in_b[i][j]); free(m->flow_in_g[i][j]);
            free(m->flow_rs_w[i][j]); free(m->flow_rs_b[i][j]); free(m->flow_rs_g[i][j]);
        }
    for (int i = 0; i < 16; i++) {
        free(m->flow_cond_v[i]); free(m->flow_cond_g[i]); free(m->flow_cond_b[i]);
        free(m->flow_pre_w[i]); free(m->flow_pre_b[i]);
        free(m->flow_post_w[i]); free(m->flow_post_b[i]);
    }
    free(m->dec_conv_pre_w); free(m->dec_conv_pre_b);
    free(m->dec_cond_w); free(m->dec_cond_b);
    for (int i = 0; i < 8; i++) { free(m->ups_w[i]); free(m->ups_g[i]); free(m->ups_b[i]); }
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 3; j++) {
            free(m->rb_conv1_w[i][j]); free(m->rb_conv1_b[i][j]); free(m->rb_conv1_g[i][j]);
            free(m->rb_conv2_w[i][j]); free(m->rb_conv2_b[i][j]); free(m->rb_conv2_g[i][j]);
        }
    for (int i = 0; i < 4; i++) { free(m->noise_w[i]); free(m->noise_b[i]); }
    free(m->emb_g);
    free(m->m_source_w); free(m->m_source_b); free(m->conv_post_w);
    pth_free(&m->pth);
}

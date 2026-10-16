/*
 * rvc_graph.c — Shared infrastructure for the RVC v2 ONNX graph emitter.
 *
 * Provides the NodeArena, emit_a(), constant emitters, and rvc_derive_params().
 * The actual graph emission is in rvc_graph_gen.c (auto-generated).
 */

#include "rvc_graph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ================================================================== */
/*  Node arena                                                         */
/* ================================================================== */

void arena_init(NodeArena *a)
{
    a->items = NULL;
    a->n = 0;
    a->cap = 0;
}

void arena_add(NodeArena *a, Buf *b)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 256;
        a->items = (Buf **)realloc(a->items, a->cap * sizeof(Buf *));
        if (!a->items) {
            fprintf(stderr, "rvc_graph: OOM allocating node array\n");
            exit(1);
        }
    }
    a->items[a->n++] = b;
}

/* ================================================================== */
/*  Node emitter                                                       */
/* ================================================================== */

void emit_a(NodeArena *a, const char *op,
            const char **in, size_t n_in,
            const char **out, size_t n_out,
            AttrSpec *attrs, size_t n_attrs)
{
    Buf *abufs = NULL;
    if (n_attrs > 0) {
        abufs = (Buf *)calloc(n_attrs, sizeof(Buf));
        if (!abufs) {
            fprintf(stderr, "rvc_graph: OOM allocating attrs\n");
            exit(1);
        }
        for (size_t i = 0; i < n_attrs; i++) {
            buf_init(&abufs[i]);
            AttrSpec *as = &attrs[i];
            switch (as->type) {
            case 'i':  onnx_emit_attr_int(&abufs[i], as->name, as->ival); break;
            case 'f':  onnx_emit_attr_float(&abufs[i], as->name, as->fval); break;
            case 'is': onnx_emit_attr_ints(&abufs[i], as->name, as->ivals, as->nvals); break;
            case 'fs': onnx_emit_attr_floats(&abufs[i], as->name, (float *)as->ivals, as->nvals); break;
            default:
                fprintf(stderr, "rvc_graph: unknown attr type '%c' for '%s'\n", as->type, as->name);
                exit(1);
            }
        }
    }

    Buf *b = (Buf *)calloc(1, sizeof(Buf));
    if (!b) {
        fprintf(stderr, "rvc_graph: OOM allocating node buffer\n");
        exit(1);
    }
    buf_init(b);

    const Buf **attr_ptrs = NULL;
    if (n_attrs > 0) {
        attr_ptrs = (const Buf **)malloc(n_attrs * sizeof(Buf *));
        for (size_t i = 0; i < n_attrs; i++) attr_ptrs[i] = &abufs[i];
    }
    onnx_emit_node(b, op, NULL, in, n_in, out, n_out, attr_ptrs, n_attrs);
    if (attr_ptrs) free(attr_ptrs);

    if (abufs) {
        for (size_t i = 0; i < n_attrs; i++)
            buf_free(&abufs[i]);
        free(abufs);
    }

    arena_add(a, b);
}

/* ================================================================== */
/*  Constant node emitters                                             */
/* ================================================================== */

/*
 * Emit a Constant node with an int64 scalar value.
 * ONNX: NodeProto(op_type="Constant", output=[name],
 *                 attribute=[name="value", t=TensorProto(data_type=INT64, raw_data=...)])
 */
void emit_const_i64(NodeArena *a, const char *out_name, int64_t val)
{
    /* Build the TensorProto for the value */
    Buf tbuf;
    buf_init(&tbuf);
    onnx_emit_tensor(&tbuf, NULL, NULL, 0, ONNX_TENSOR_INT64,
                     (const uint8_t *)&val, sizeof(val));

    /* Build the AttributeProto: name="value", type=TENSOR, t=<TensorProto> */
    Buf abuf;
    buf_init(&abuf);
    pb_string(&abuf, 1, "value");
    pb_message(&abuf, 5, &tbuf);  /* field 5 = t (TensorProto) */
    pb_uint32(&abuf, 20, ONNX_ATTR_TENSOR);  /* type=4 */

    /* Emit the Constant node */
    Buf *b = (Buf *)calloc(1, sizeof(Buf));
    buf_init(b);
    const char *out[] = { out_name };
    const Buf *attr_ptrs[] = { &abuf };
    onnx_emit_node(b, "Constant", NULL, NULL, 0, out, 1, attr_ptrs, 1);

    buf_free(&tbuf);
    buf_free(&abuf);
    arena_add(a, b);
}

/*
 * Emit a Constant node with a float32 scalar value.
 */
void emit_const_f32(NodeArena *a, const char *out_name, float val)
{
    Buf tbuf;
    buf_init(&tbuf);
    onnx_emit_tensor(&tbuf, NULL, NULL, 0, ONNX_TENSOR_FLOAT,
                     (const uint8_t *)&val, sizeof(val));

    Buf abuf;
    buf_init(&abuf);
    pb_string(&abuf, 1, "value");
    pb_message(&abuf, 5, &tbuf);
    pb_uint32(&abuf, 20, ONNX_ATTR_TENSOR);

    Buf *b = (Buf *)calloc(1, sizeof(Buf));
    buf_init(b);
    const char *out[] = { out_name };
    const Buf *attr_ptrs[] = { &abuf };
    onnx_emit_node(b, "Constant", NULL, NULL, 0, out, 1, attr_ptrs, 1);

    buf_free(&tbuf);
    buf_free(&abuf);
    arena_add(a, b);
}

/*
 * Emit a Constant node with an int64 vector of 1 element.
 */
void emit_const_i64v1(NodeArena *a, const char *out_name, int64_t val)
{
    Buf tbuf;
    buf_init(&tbuf);
    int64_t dim = 1;
    onnx_emit_tensor(&tbuf, NULL, &dim, 1, ONNX_TENSOR_INT64,
                     (const uint8_t *)&val, sizeof(val));

    Buf abuf;
    buf_init(&abuf);
    pb_string(&abuf, 1, "value");
    pb_message(&abuf, 5, &tbuf);
    pb_uint32(&abuf, 20, ONNX_ATTR_TENSOR);

    Buf *b = (Buf *)calloc(1, sizeof(Buf));
    buf_init(b);
    const char *out[] = { out_name };
    const Buf *attr_ptrs[] = { &abuf };
    onnx_emit_node(b, "Constant", NULL, NULL, 0, out, 1, attr_ptrs, 1);

    buf_free(&tbuf);
    buf_free(&abuf);
    arena_add(a, b);
}

/* ================================================================== */
/*  Parameter derivation                                               */
/* ================================================================== */

int rvc_derive_params(RvcParams *p,
                      const int64_t *config, size_t config_len,
                      int hub_dim)
{
    if (!p) return -1;

    /* Defaults for RVC v2 */
    p->spec_channels     = 1025;
    p->segment_size      = 32;
    p->inter_channels    = 192;
    p->hidden_channels   = 192;
    p->filter_channels   = 768;
    p->n_heads           = 2;
    p->n_layers          = 6;
    p->kernel_size       = 3;
    p->resblock          = 1;
    p->resblock_kernel_sizes[0] = 3;
    p->resblock_kernel_sizes[1] = 7;
    p->resblock_kernel_sizes[2] = 11;
    p->resblock_dilations[0][0] = 1; p->resblock_dilations[0][1] = 3; p->resblock_dilations[0][2] = 5;
    p->resblock_dilations[1][0] = 1; p->resblock_dilations[1][1] = 3; p->resblock_dilations[1][2] = 5;
    p->resblock_dilations[2][0] = 1; p->resblock_dilations[2][1] = 3; p->resblock_dilations[2][2] = 5;
    p->n_upsamples = 4;
    p->upsample_rates[0] = 10;
    p->upsample_rates[1] = 10;
    p->upsample_rates[2] = 2;
    p->upsample_rates[3] = 2;
    p->upsample_initial_channel = 512;
    p->upsample_kernel_sizes[0] = 16;
    p->upsample_kernel_sizes[1] = 16;
    p->upsample_kernel_sizes[2] = 4;
    p->upsample_kernel_sizes[3] = 4;
    p->spk_embed_dim = 109;
    p->gin_channels = 256;
    p->sr = 48000;

    p->hub_dim = hub_dim;
    p->k_channels = p->hidden_channels / p->n_heads;
    p->window_size = 10;
    p->n_flows = 4;
    p->flow_indices[0] = 0;
    p->flow_indices[1] = 2;
    p->flow_indices[2] = 4;
    p->flow_indices[3] = 6;
    p->n_resblocks_per_stage = 3;
    p->upp = p->upsample_rates[0] * p->upsample_rates[1] *
             p->upsample_rates[2] * p->upsample_rates[3];
    p->sine_amp = 0.1f;
    p->noise_std = 0.003f;
    p->harmonic_num = 0;

    (void)config; (void)config_len;
    return 0;
}

/* ================================================================== */
/*  Cleanup                                                            */
/* ================================================================== */

void rvc_free_nodes(Buf **nodes, size_t n_nodes)
{
    for (size_t i = 0; i < n_nodes; i++)
        buf_free(&nodes[i]);
    free(nodes);
}

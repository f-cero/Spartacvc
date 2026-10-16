/*
 * rvc_graph.h — RVC v2 ONNX graph emitter.
 *
 * Emits the complete RVC v2 voice conversion graph as ONNX NodeProto
 * messages. The graph implements the forward pass of
 * SynthesizerTrnMsNSFsidM from models_onnx.py.
 *
 * All architecture parameters are derived from the .pth config and
 * tensor shapes — nothing is hardcoded to a specific model.
 *
 * Constants are emitted as Constant nodes (NodeProto with op_type="Constant"
 * and a "value" TensorProto attribute). Weight tensors are emitted as
 * initializers (TensorProto in graph.initializer).
 */

#ifndef SPCV_RVC_GRAPH_H
#define SPCV_RVC_GRAPH_H

#include "onnx_emit.h"

/* ================================================================== */
/*  Node arena (shared infrastructure)                                */
/* ================================================================== */

typedef struct {
    Buf    **items;
    size_t   n;
    size_t   cap;
} NodeArena;

void    arena_init(NodeArena *a);
void    arena_add(NodeArena *a, Buf *b);

/*
 * Attribute specification for emit_a().
 * type: 'i'=int, 'f'=float, 'is'=int-list, 'fs'=float-list
 */
typedef struct {
    const char *name;
    char type;
    int64_t ival;
    float   fval;
    const int64_t *ivals;
    size_t nvals;
} AttrSpec;

/*
 * Generic node emitter.
 */
void emit_a(NodeArena *a, const char *op,
            const char **in, size_t n_in,
            const char **out, size_t n_out,
            AttrSpec *attrs, size_t n_attrs);

/*
 * Emit a constant scalar as a Constant node.
 * These are emitted as NodeProto with op_type="Constant" and a
 * "value" attribute containing the TensorProto.
 */
void emit_const_i64(NodeArena *a, const char *out_name, int64_t val);
void emit_const_f32(NodeArena *a, const char *out_name, float val);
void emit_const_i64v1(NodeArena *a, const char *out_name, int64_t val);

/* ================================================================== */
/*  Architecture parameters                                           */
/* ================================================================== */

typedef struct {
    int    spec_channels;
    int    segment_size;
    int    inter_channels;
    int    hidden_channels;
    int    filter_channels;
    int    n_heads;
    int    n_layers;
    int    kernel_size;
    int    resblock;
    int    resblock_kernel_sizes[3];
    int    resblock_dilations[3][3];
    int    n_upsamples;
    int    upsample_rates[4];
    int    upsample_initial_channel;
    int    upsample_kernel_sizes[4];
    int    spk_embed_dim;
    int    gin_channels;
    int    sr;
    int    hub_dim;
    int    k_channels;
    int    window_size;
    int    n_flows;
    int    flow_indices[4];
    int    n_resblocks_per_stage;
    int    upp;
    float  sine_amp;
    float  noise_std;
    int    harmonic_num;
} RvcParams;

int rvc_derive_params(RvcParams *p,
                      const int64_t *config, size_t config_len,
                      int hub_dim);

typedef struct {
    const char *name;
    int64_t    *dims;
    int         ndim;
    float      *data;
    size_t      nbytes;
} RvcWeight;

int rvc_emit_graph(const RvcParams *p,
                   const RvcWeight *weights, size_t n_weights,
                   Buf ***nodes_out, size_t *n_nodes_out);

void rvc_free_nodes(Buf **nodes, size_t n_nodes);

#endif /* SPCV_RVC_GRAPH_H */

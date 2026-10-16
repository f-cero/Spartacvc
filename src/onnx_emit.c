/*
 * onnx_emit.c — Minimal ONNX protobuf writer.
 *
 * Implements the protobuf wire format directly (no protobuf library).
 * Emits ONNX ModelProto, GraphProto, NodeProto, TensorProto,
 * AttributeProto, and ValueInfoProto messages.
 *
 * All multi-byte integers are little-endian (x86/x64 native).
 * Varints use the standard protobuf 7-bits-per-byte encoding.
 */

#include "onnx_emit.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ================================================================== */
/*  Growable byte buffer                                              */
/* ================================================================== */

void buf_init(Buf *b)
{
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

void buf_free(Buf *b)
{
    free(b->data);
    b->data = NULL;
    b->len  = 0;
    b->cap  = 0;
}

void buf_reserve(Buf *b, size_t extra)
{
    if (extra > (size_t)1 << 30) {
        fprintf(stderr, "onnx_emit: absurd extra=%zu (0x%zx) buf: len=%zu cap=%zu data=%p\n",
                extra, extra, b->len, b->cap, b->data);
        fflush(stderr);
        exit(1);
    }
    size_t need = b->len + extra;
    if (need < b->len) { /* overflow */
        fprintf(stderr, "onnx_emit: size overflow len=%zu extra=%zu\n", b->len, extra);
        exit(1);
    }
    if (need <= b->cap) return;
    size_t newcap = b->cap ? b->cap : 256;
    while (newcap < need)
        newcap *= 2;
    b->data = (uint8_t *)realloc(b->data, newcap);
    if (!b->data) {
        fprintf(stderr, "onnx_emit: out of memory (%zu bytes)\n", newcap);
        exit(1);
    }
    b->cap = newcap;
}

void buf_put(Buf *b, const void *p, size_t n)
{
    buf_reserve(b, n);
    memcpy(b->data + b->len, p, n);
    b->len += n;
}

void buf_put_u8(Buf *b, uint8_t v)
{
    buf_reserve(b, 1);
    b->data[b->len++] = v;
}

void buf_put_u32(Buf *b, uint32_t v)
{
    uint8_t tmp[4] = {
        (uint8_t)(v & 0xff),
        (uint8_t)((v >> 8) & 0xff),
        (uint8_t)((v >> 16) & 0xff),
        (uint8_t)((v >> 24) & 0xff),
    };
    buf_put(b, tmp, 4);
}

void buf_put_u64(Buf *b, uint64_t v)
{
    uint8_t tmp[8];
    for (int i = 0; i < 8; i++)
        tmp[i] = (uint8_t)((v >> (i * 8)) & 0xff);
    buf_put(b, tmp, 8);
}

/* ================================================================== */
/*  Protobuf primitives                                               */
/* ================================================================== */

/*
 * Encode an unsigned integer as a protobuf varint.
 * 7 bits per byte, MSB = continuation flag.
 */
void pb_varint(Buf *b, uint64_t v)
{
    do {
        uint8_t byte = (uint8_t)(v & 0x7f);
        v >>= 7;
        if (v) byte |= 0x80;
        buf_put_u8(b, byte);
    } while (v);
}

/*
 * Emit a protobuf field tag: (field_number << 3) | wire_type.
 */
void pb_tag(Buf *b, uint32_t field, uint32_t wire_type)
{
    pb_varint(b, ((uint64_t)field << 3) | wire_type);
}

/*
 * Emit a varint field (wire type 0).
 */
void pb_uint32(Buf *b, uint32_t field, uint32_t v)
{
    pb_tag(b, field, 0);
    pb_varint(b, v);
}

/*
 * Emit an int64 field (wire type 0, varint encoding).
 * Negative values are encoded as 10-byte varints per protobuf spec.
 */
void pb_int64(Buf *b, uint32_t field, int64_t v)
{
    pb_tag(b, field, 0);
    pb_varint(b, (uint64_t)v);
}

/*
 * Emit a 32-bit float field (wire type 5).
 */
void pb_float(Buf *b, uint32_t field, float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    pb_tag(b, field, 5);
    buf_put_u32(b, u);
}

/*
 * Emit a string field (wire type 2, length-delimited).
 */
void pb_string(Buf *b, uint32_t field, const char *s)
{
    size_t len = strlen(s);
    pb_tag(b, field, 2);
    pb_varint(b, len);
    buf_put(b, s, len);
}

/*
 * Emit a bytes field (wire type 2, length-delimited).
 */
void pb_bytes(Buf *b, uint32_t field, const uint8_t *p, size_t n)
{
    pb_tag(b, field, 2);
    pb_varint(b, n);
    buf_put(b, p, n);
}

/*
 * Emit a nested message field (wire type 2, length-delimited).
 * The message content is already serialized in `msg`.
 */
void pb_message(Buf *b, uint32_t field, const Buf *msg)
{
    pb_tag(b, field, 2);
    pb_varint(b, msg->len);
    buf_put(b, msg->data, msg->len);
}

/* ================================================================== */
/*  ONNX TensorProto                                                  */
/* ================================================================== */

/*
 * TensorProto protobuf fields:
 *   1  (dims):      repeated int64
 *   2  (data_type): int32
 *   4  (name):      string
 *   6  (float_data): repeated float   (not used, we use raw_data)
 *   7  (int32_data): repeated int32   (not used)
 *   8  (string_data):repeated bytes   (not used)
 *   9  (int64_data): repeated int64   (not used)
 *   13 (raw_data):  bytes
 */
void onnx_emit_tensor(Buf *out,
                      const char *name,
                      const int64_t *dims, size_t ndim,
                      int elem_type,
                      const uint8_t *data, size_t nbytes)
{
    /* dims: repeated int64, field 1 */
    for (size_t i = 0; i < ndim; i++)
        pb_int64(out, 1, dims[i]);

    /* data_type: int32, field 2 */
    pb_uint32(out, 2, (uint32_t)elem_type);

    /* name: string, field 8 */
    if (name && name[0])
        pb_string(out, 8, name);

    /* raw_data: bytes, field 9 */
    if (data && nbytes > 0)
        pb_bytes(out, 9, data, nbytes);
}

/* ================================================================== */
/*  ONNX AttributeProto                                               */
/* ================================================================== */

/*
 * AttributeProto protobuf fields:
 *   1  (name):   string
 *   2  (f):      float
 *   3  (i):      int64
 *   4  (s):      bytes
 *   5  (t):      TensorProto
 *   6  (g):      GraphProto
 *   7  (floats): repeated float
 *   8  (ints):   repeated int64
 *   9  (tensors):repeated TensorProto
 *   10 (graphs): repeated GraphProto
 *   20 (type):   enum (AttributeType)
 */

void onnx_emit_attr_int(Buf *out, const char *name, int64_t v)
{
    pb_string(out, 1, name);
    pb_int64(out, 3, v);
    pb_uint32(out, 20, ONNX_ATTR_INT);
}

void onnx_emit_attr_float(Buf *out, const char *name, float v)
{
    pb_string(out, 1, name);
    pb_float(out, 2, v);
    pb_uint32(out, 20, ONNX_ATTR_FLOAT);
}

void onnx_emit_attr_ints(Buf *out, const char *name,
                         const int64_t *vals, size_t n)
{
    pb_string(out, 1, name);
    for (size_t i = 0; i < n; i++)
        pb_int64(out, 8, vals[i]);
    pb_uint32(out, 20, ONNX_ATTR_INTS); /* type=7 is INTS */
}

void onnx_emit_attr_floats(Buf *out, const char *name,
                           const float *vals, size_t n)
{
    pb_string(out, 1, name);
    for (size_t i = 0; i < n; i++)
        pb_float(out, 7, vals[i]);
    pb_uint32(out, 20, ONNX_ATTR_FLOATS);
}

void onnx_emit_attr_string(Buf *out, const char *name, const char *v)
{
    pb_string(out, 1, name);
    pb_bytes(out, 4, (const uint8_t *)v, strlen(v));
    pb_uint32(out, 20, ONNX_ATTR_STRING);
}

/* ================================================================== */
/*  ONNX NodeProto                                                    */
/* ================================================================== */

/*
 * NodeProto protobuf fields:
 *   1 (input):    repeated string
 *   2 (output):   repeated string
 *   3 (name):     string
 *   4 (op_type):  string
 *   5 (attribute):repeated AttributeProto
 *   7 (doc_string): string
 *   8 (domain):   string
 */
void onnx_emit_node(Buf *out,
                    const char *op_type,
                    const char *name,
                    const char **inputs, size_t n_inputs,
                    const char **outputs, size_t n_outputs,
                    const Buf **attrs, size_t n_attrs)
{
    /* inputs: repeated string, field 1 */
    for (size_t i = 0; i < n_inputs; i++)
        pb_string(out, 1, inputs[i]);

    /* outputs: repeated string, field 2 */
    for (size_t i = 0; i < n_outputs; i++)
        pb_string(out, 2, outputs[i]);

    /* name: string, field 3 */
    if (name && name[0])
        pb_string(out, 3, name);

    /* op_type: string, field 4 */
    pb_string(out, 4, op_type);

    /* attributes: repeated AttributeProto, field 5 */
    for (size_t i = 0; i < n_attrs; i++)
        pb_message(out, 5, attrs[i]);
}

/*
 * NodeProto emitter with explicit string lengths (for non-null-terminated strings).
 * input_lens / output_lens can be NULL if the strings are null-terminated.
 */
void onnx_emit_node_len(Buf *out,
                        const char *op_type, size_t op_len,
                        const char **inputs, const size_t *input_lens, size_t n_inputs,
                        const char **outputs, const size_t *output_lens, size_t n_outputs,
                        const Buf **attrs, size_t n_attrs)
{
    for (size_t i = 0; i < n_inputs; i++) {
        if (input_lens)
            pb_bytes(out, 1, (const uint8_t *)inputs[i], input_lens[i]);
        else
            pb_string(out, 1, inputs[i]);
    }
    for (size_t i = 0; i < n_outputs; i++) {
        if (output_lens)
            pb_bytes(out, 2, (const uint8_t *)outputs[i], output_lens[i]);
        else
            pb_string(out, 2, outputs[i]);
    }
    pb_bytes(out, 4, (const uint8_t *)op_type, op_len);
    for (size_t i = 0; i < n_attrs; i++)
        pb_message(out, 5, attrs[i]);
}

/* ================================================================== */
/*  ONNX ValueInfoProto                                               */
/* ================================================================== */

/*
 * ValueInfoProto:
 *   1 (name): string
 *   2 (type): TypeProto
 *
 * TypeProto:
 *   1 (tensor_type): TensorTypeProto
 *
 * TensorTypeProto:
 *   1 (elem_type): int32
 *   2 (shape):     TensorShapeProto
 *
 * TensorShapeProto:
 *   1 (dim): repeated Dimension
 *
 * Dimension:
 *   1 (dim_value): int64
 *   2 (dim_param): string
 */
void onnx_emit_value_info(Buf *out,
                          const char *name,
                          int elem_type,
                          const int64_t *dims,
                          const char **dim_params,
                          size_t ndim)
{
    /* Build Dimension entries */
    Buf shape;
    buf_init(&shape);
    for (size_t i = 0; i < ndim; i++) {
        Buf dim;
        buf_init(&dim);
        if (dim_params && dim_params[i]) {
            pb_string(&dim, 2, dim_params[i]);
        } else {
            pb_int64(&dim, 1, dims ? dims[i] : 0);
        }
        pb_message(&shape, 1, &dim);
        buf_free(&dim);
    }

    /* Build TensorTypeProto */
    Buf ttype;
    buf_init(&ttype);
    pb_uint32(&ttype, 1, (uint32_t)elem_type);
    pb_message(&ttype, 2, &shape);
    buf_free(&shape);

    /* Build TypeProto */
    Buf type;
    buf_init(&type);
    pb_message(&type, 1, &ttype);
    buf_free(&ttype);

    /* Emit ValueInfoProto */
    pb_string(out, 1, name);
    pb_message(out, 2, &type);
    buf_free(&type);
}

/* ================================================================== */
/*  ONNX GraphProto                                                   */
/* ================================================================== */

/*
 * GraphProto protobuf fields:
 *   1  (name):        string
 *   2  (node):        repeated NodeProto
 *   5  (initializer): repeated TensorProto
 *   6  (doc_string):  string
 *   11 (input):       repeated ValueInfoProto
 *   12 (output):      repeated ValueInfoProto
 *   13 (value_info):  repeated ValueInfoProto
 */
void onnx_build_graph(Buf *out,
                      const char *name,
                      const Buf **nodes, size_t n_nodes,
                      const Buf **inits, size_t n_inits,
                      const Buf **inputs, size_t n_inputs,
                      const Buf **outputs, size_t n_outputs)
{
    /* nodes: repeated NodeProto, field 1 */
    for (size_t i = 0; i < n_nodes; i++)
        pb_message(out, 1, nodes[i]);

    /* name: string, field 2 */
    if (name && name[0])
        pb_string(out, 2, name);

    /* initializers: repeated TensorProto, field 5 */
    for (size_t i = 0; i < n_inits; i++)
        pb_message(out, 5, inits[i]);

    /* inputs: repeated ValueInfoProto, field 11 */
    for (size_t i = 0; i < n_inputs; i++)
        pb_message(out, 11, inputs[i]);

    /* outputs: repeated ValueInfoProto, field 12 */
    for (size_t i = 0; i < n_outputs; i++)
        pb_message(out, 12, outputs[i]);
}

/* ================================================================== */
/*  ONNX ModelProto                                                   */
/* ================================================================== */

/*
 * ModelProto protobuf fields:
 *   1  (ir_version):       int64
 *   2  (producer_name):    string
 *   3  (producer_version): string
 *   7  (graph):            GraphProto
 *   8  (opset_import):     repeated OperatorSetIdProto
 *
 * OperatorSetIdProto:
 *   1 (domain):  string
 *   2 (version): int64
 */
void onnx_emit_model(Buf *out,
                     const Buf *graph,
                     int ir_version,
                     int opset_version,
                     const char *producer)
{
    /* ir_version: int64, field 1 */
    pb_int64(out, 1, ir_version);

    /* producer_name: string, field 2 */
    if (producer)
        pb_string(out, 2, producer);

    /* producer_version: string, field 3 */
    pb_string(out, 3, "1.0");

    /* graph: GraphProto, field 7 */
    pb_message(out, 7, graph);

    /* opset_import: repeated OperatorSetIdProto, field 8 */
    {
        Buf opset;
        buf_init(&opset);
        pb_string(&opset, 1, "");  /* default domain */
        pb_int64(&opset, 2, opset_version);
        pb_message(out, 8, &opset);
        buf_free(&opset);
    }
}

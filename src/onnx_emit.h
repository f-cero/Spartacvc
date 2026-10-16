/*
 * onnx_emit.h — Minimal ONNX protobuf writer for pth2onnx.
 *
 * Emits ONNX ModelProto / GraphProto / NodeProto / TensorProto /
 * AttributeProto / ValueInfoProto messages using the protobuf wire
 * format directly. No external protobuf library required.
 *
 * All functions append to a growable byte buffer (Buf).
 *
 * Wire format reference (ONNX protobuf schema):
 *   tag = (field_number << 3) | wire_type
 *   wire_type: 0=varint, 1=64-bit, 2=length-delimited, 5=32-bit
 */

#ifndef SPCV_ONNX_EMIT_H
#define SPCV_ONNX_EMIT_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Growable byte buffer                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} Buf;

void buf_init(Buf *b);
void buf_free(Buf *b);
void buf_reserve(Buf *b, size_t extra);
void buf_put(Buf *b, const void *p, size_t n);
void buf_put_u8(Buf *b, uint8_t v);
void buf_put_u32(Buf *b, uint32_t v);  /* little-endian */
void buf_put_u64(Buf *b, uint64_t v);  /* little-endian */

/* ------------------------------------------------------------------ */
/*  Protobuf primitives                                                */
/* ------------------------------------------------------------------ */

void pb_varint(Buf *b, uint64_t v);
void pb_tag(Buf *b, uint32_t field, uint32_t wire_type);

/* Scalar field emitters */
void pb_uint32(Buf *b, uint32_t field, uint32_t v);   /* varint */
void pb_int64(Buf *b, uint32_t field, int64_t v);      /* varint */
void pb_float(Buf *b, uint32_t field, float f);        /* 32-bit */
void pb_string(Buf *b, uint32_t field, const char *s); /* len-delim */
void pb_bytes(Buf *b, uint32_t field, const uint8_t *p, size_t n); /* len-delim */

/* Nested message emitter */
void pb_message(Buf *b, uint32_t field, const Buf *msg);

/* ------------------------------------------------------------------ */
/*  ONNX message emitters                                              */
/* ------------------------------------------------------------------ */

/*
 * TensorProto data type codes (onnx.proto).
 */
enum {
    ONNX_TENSOR_FLOAT    = 1,
    ONNX_TENSOR_UINT8    = 2,
    ONNX_TENSOR_INT8     = 3,
    ONNX_TENSOR_INT32    = 6,
    ONNX_TENSOR_INT64    = 7,
    ONNX_TENSOR_BOOL     = 9,
    ONNX_TENSOR_FLOAT16  = 10,
    ONNX_TENSOR_DOUBLE   = 11,
};

/*
 * AttributeProto type codes.
 */
enum {
    ONNX_ATTR_FLOAT  = 1,
    ONNX_ATTR_INT    = 2,
    ONNX_ATTR_STRING = 3,
    ONNX_ATTR_TENSOR = 4,
    ONNX_ATTR_GRAPH  = 5,
    ONNX_ATTR_FLOATS = 6,
    ONNX_ATTR_INTS   = 7,
    ONNX_ATTR_STRINGS= 8,
};

/*
 * Emit a single TensorProto into `out`.
 *
 *   name     — initializer name (e.g. "dec.conv_pre.weight")
 *   dims     — tensor dimensions
 *   ndim     — number of dimensions
 *   elem_type— ONNX_TENSOR_* code
 *   data     — raw little-endian tensor data (float32, int64, etc.)
 *   nbytes   — total byte length of `data`
 *
 * TensorProto fields emitted:
 *   1 (dims):      repeated int64
 *   2 (data_type): int32
 *   4 (name):      string
 *   13 (raw_data): bytes
 */
void onnx_emit_tensor(Buf *out,
                      const char *name,
                      const int64_t *dims, size_t ndim,
                      int elem_type,
                      const uint8_t *data, size_t nbytes);

/*
 * Emit a single AttributeProto into `out`.
 *
 * Supports: int, float, int-list, float-list, string, tensor.
 * The `type` parameter is the ONNX_ATTR_* code.
 */
void onnx_emit_attr_int(Buf *out, const char *name, int64_t v);
void onnx_emit_attr_float(Buf *out, const char *name, float v);
void onnx_emit_attr_ints(Buf *out, const char *name,
                         const int64_t *vals, size_t n);
void onnx_emit_attr_floats(Buf *out, const char *name,
                           const float *vals, size_t n);
void onnx_emit_attr_string(Buf *out, const char *name, const char *v);

/*
 * Emit a single NodeProto into `out`.
 *
 *   op_type  — operator name (e.g. "Conv", "MatMul", "LeakyRelu")
 *   name     — node name (e.g. "/dec/conv_pre")
 *   inputs   — input tensor names
 *   n_inputs — number of inputs
 *   outputs  — output tensor names
 *   n_outputs— number of outputs
 *   attrs    — array of pre-built AttributeProto buffers
 *   n_attrs  — number of attributes
 */
void onnx_emit_node(Buf *out,
                    const char *op_type,
                    const char *name,
                    const char **inputs, size_t n_inputs,
                    const char **outputs, size_t n_outputs,
                    const Buf **attrs, size_t n_attrs);

void onnx_emit_node_len(Buf *out,
                        const char *op_type, size_t op_len,
                        const char **inputs, const size_t *input_lens, size_t n_inputs,
                        const char **outputs, const size_t *output_lens, size_t n_outputs,
                        const Buf **attrs, size_t n_attrs);

/*
 * Emit a single ValueInfoProto into `out`.
 *
 *   name      — tensor name
 *   elem_type — ONNX_TENSOR_* code
 *   dims      — fixed dimensions (use 0 for dynamic)
 *   dim_params— symbolic names for dynamic dims (e.g. "T"), NULL for fixed
 *   ndim      — number of dimensions
 */
void onnx_emit_value_info(Buf *out,
                          const char *name,
                          int elem_type,
                          const int64_t *dims,
                          const char **dim_params,
                          size_t ndim);

/*
 * Emit the full ModelProto into `out`.
 *
 *   graph         — pre-built GraphProto buffer
 *   ir_version    — ONNX IR version (7)
 *   opset_version — ai.onnx opset version (13)
 *   producer      — producer name string
 */
void onnx_emit_model(Buf *out,
                     const Buf *graph,
                     int ir_version,
                     int opset_version,
                     const char *producer);

/*
 * Convenience: build a complete GraphProto from nodes + initializers
 * + inputs + outputs.
 *
 * All arrays are pre-built buffers.
 */
void onnx_build_graph(Buf *out,
                      const char *name,
                      const Buf **nodes, size_t n_nodes,
                      const Buf **inits, size_t n_inits,
                      const Buf **inputs, size_t n_inputs,
                      const Buf **outputs, size_t n_outputs);

#endif /* SPCV_ONNX_EMIT_H */

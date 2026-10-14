/**
 * @file    onnx.h
 * @brief   ONNX Runtime C API wrapper for spartacvc.
 *
 * Provides a thin, documented layer over the ONNX Runtime C API.
 * All ORT calls go through this module — no other file touches ORT directly.
 *
 * Pipeline role:
 *   hubert.c  ->  spcv_onnx_run()  ->  HuBERT embeddings
 *   f0.c      ->  spcv_onnx_run()  ->  RMVPE f0 output
 *   voice.c   ->  spcv_onnx_run()  ->  decoder PCM output
 *
 * Debug:
 *   Compile with -DSPCV_DEBUG AND launch with --debug (or SPCV_DEBUG=1)
 *   to emit trace lines to spcv_trace.log and stderr.
 */
#ifndef SPCV_ONNX_H
#define SPCV_ONNX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Opaque ORT types (forward-declared; defined in onnxruntime_c_api) */
/* ------------------------------------------------------------------ */
typedef struct OrtEnv              spcv_ort_env;
typedef struct OrtSessionOptions   spcv_ort_opts;
typedef struct OrtSession          spcv_ort_session;
typedef struct OrtMemoryInfo      spcv_ort_meminfo;
typedef struct OrtValue           spcv_ort_value;
typedef struct OrtAllocator       spcv_ort_alloc;

/* ------------------------------------------------------------------ */
/*  Error handling                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief  ORT error codes (subset we care about).
 */
typedef enum {
    SPCV_ONNX_OK            = 0,
    SPCV_ONNX_ERR_LOAD      = 1,   /* model file not found / corrupt      */
    SPCV_ONNX_ERR_INPUT     = 2,   /* bad input shape or type             */
    SPCV_ONNX_ERR_RUN       = 3,   /* Run() failed                        */
    SPCV_ONNX_ERR_ALLOC     = 4,   /* memory allocation failure           */
    SPCV_ONNX_ERR_API       = 5,   /* ORT API version mismatch            */
    SPCV_ONNX_ERR_UNKNOWN   = 99
} spcv_onnx_status;

/**
 * @brief  Human-readable error string for a status code.
 * @param  st  status code
 * @return  static string, do not free
 */
const char *spcv_onnx_status_str(spcv_onnx_status st);

/* ------------------------------------------------------------------ */
/*  Tensor descriptor (our own, ORT-free)                              */
/* ------------------------------------------------------------------ */

/**
 * @brief  Describes one input or output tensor of a model.
 */
typedef struct {
    char      name[256];          /* tensor name (e.g. "source")          */
    int       ndim;               /* number of dimensions                 */
    int64_t   dims[8];            /* shape (up to 8 dims)                 */
    int       elem_type;          /* ONNX enum: 1=float, 7=int64, etc.   */
    size_t    byte_size;          /* total bytes (product of dims * esz)  */
} spcv_tensor_desc;

/* ------------------------------------------------------------------ */
/*  Session                                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief  An ONNX inference session.
 *
 * Holds the ORT session, input/output descriptors, and a reference
 * to the shared ORT environment.
 */
typedef struct spcv_onnx_session {
    /* ORT objects (opaque pointers) */
    void    *ort_session;         /* OrtSession*                          */
    void    *ort_meminfo;        /* OrtMemoryInfo* (CPU)                 */
    void    *ort_alloc;          /* OrtAllocator*                        */

    /* Descriptors */
    int              n_inputs;
    int              n_outputs;
    spcv_tensor_desc inputs[8];
    spcv_tensor_desc outputs[8];

    /* Shared env (owned by the first session that creates it) */
    int              owns_env;
    void            *ort_env;     /* OrtEnv* (shared)                   */
} spcv_onnx_session;

/**
 * @brief  Execution provider selection.
 */
typedef enum {
    SPCV_EP_AUTO     = 0,  /* try DML, fall back to CPU          */
    SPCV_EP_DML      = 1,  /* DirectML (Windows D3D12 GPU)       */
    SPCV_EP_CPU      = 2   /* CPU only                            */
} spcv_ep;

/**
 * @brief  Set the preferred execution provider (call before spcv_onnx_load).
 *
 * @param  ep  execution provider (SPCV_EP_AUTO / DML / CPU)
 */
void spcv_onnx_set_ep(spcv_ep ep);

/**
 * @brief  Global ORT environment (created once, shared across sessions).
 *
 * @return  0 on success, SPCV_ONNX_ERR_API on failure
 */
int spcv_onnx_init(void);

/**
 * @brief  Release the global ORT environment.
 */
void spcv_onnx_shutdown(void);

/**
 * @brief  Load an ONNX model and create a session.
 *
 * @param  path      UTF-8 path to .onnx file
 * @param  sess_out  [out] session handle (caller frees with spcv_onnx_free)
 * @return  SPCV_ONNX_OK or error code
 *
 * Traces (debug): model path, input/output names and shapes.
 */
int spcv_onnx_load(const char *path, spcv_onnx_session **sess_out);

/**
 * @brief  Free a session and its descriptors.
 *
 * Does NOT free the shared ORT env (call spcv_onnx_shutdown for that).
 */
void spcv_onnx_free(spcv_onnx_session *sess);

/* ------------------------------------------------------------------ */
/*  Inference                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief  A single input tensor: name + raw data pointer + shape.
 *
 * The caller owns `data` and must keep it alive for the duration of Run().
 */
typedef struct {
    const char  *name;            /* must match one of sess->inputs[].name */
    const void  *data;            /* raw buffer (float32 / int64 / etc.)  */
    int          ndim;
    int64_t      dims[8];
    int          elem_type;       /* ONNX tensor element type enum        */
} spcv_onnx_input;

/**
 * @brief  A single output tensor: where to receive the result.
 *
 * Two modes:
 *   - `buf == NULL`:  ORT allocates, caller reads via spcv_onnx_output_data()
 *                     and frees with spcv_onnx_free_output().
 *   - `buf != NULL`:  caller provides a pre-allocated buffer of sufficient
 *                     size (byte_size from sess->outputs[]).
 */
typedef struct {
    const char  *name;            /* must match one of sess->outputs[].name */
    void        *buf;             /* NULL = ORT allocates, else caller buf  */
    size_t       buf_size;        /* bytes available in buf (if provided)   */
    /* filled after Run(): */
    void        *ort_value;       /* OrtValue* (valid until free_output)   */
} spcv_onnx_output;

/**
 * @brief  Run inference.
 *
 * @param  sess      loaded session
 * @param  inputs    array of input tensors
 * @param  n_inputs  number of inputs
 * @param  outputs   array of output slots
 * @param  n_outputs number of outputs
 * @return  SPCV_ONNX_OK or error code
 *
 * Traces (debug): input shapes, timing, output shapes.
 */
int spcv_onnx_run(spcv_onnx_session *sess,
                  const spcv_onnx_input  *inputs,  int n_inputs,
                  spcv_onnx_output     *outputs, int n_outputs);

/**
 * @brief  Get a pointer to the raw data of an output tensor.
 *
 * @param  out  output slot (after successful spcv_onnx_run)
 * @return  pointer to raw data (float32*, int64*, etc.) or NULL
 */
const void *spcv_onnx_output_data(const spcv_onnx_output *out);

/**
 * @brief  Get the actual shape of an output tensor.
 *
 * @param  out    output slot
 * @param  ndim   [out] number of dims
 * @param  dims   [out] shape array (caller provides storage for up to 8)
 */
void spcv_onnx_output_shape(const spcv_onnx_output *out,
                            int *ndim, int64_t *dims);

/**
 * @brief  Free an ORT-allocated output tensor.
 *
 * Call once per output after you are done reading the data.
 */
void spcv_onnx_free_output(spcv_onnx_output *out);

#ifdef __cplusplus
}
#endif

#endif /* SPCV_ONNX_H */

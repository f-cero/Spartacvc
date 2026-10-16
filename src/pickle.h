// pickle.h — minimal Python pickle protocol 2 reader for .pth files
#ifndef SPARTACVC_PICKLE_H
#define SPARTACVC_PICKLE_H

#include <stdint.h>
#include <stddef.h>

// --- Value types ---
typedef enum {
    PKV_INT,
    PKV_STR,
    PKV_BOOL,
    PKV_TUPLE,
    PKV_LIST,
    PKV_DICT,
    PKV_FALSE,
    PKV_STORAGE,   // tensor storage reference
    PKV_TENSOR,     // rebuilt tensor
    PKV_REF,        // callable reference (GLOBAL)
} pkv_type_t;

typedef struct pkv pkv_t;

struct pkv {
    pkv_type_t type;
    union {
        int64_t  i;
        int      bool_val;
        struct { char *data; size_t len; } str;
        struct { pkv_t **items; int count; } tuple;
        struct { pkv_t **items; int count; } list;
        struct { pkv_t **keys; pkv_t **vals; int count; } dict;
        // STORAGE: tensor data reference
        struct {
            char     dtype[32];     // "Half", "Float", "Int", "Long"
            char     data_file[16]; // "0", "1", ...
            int64_t  size_bytes;
        } storage;
        // TENSOR: full tensor record
        struct {
            char     name[256];
            int      data_file;
            int64_t  size_bytes;
            int      ndim;
            int64_t  shape[8];
            int      dtype;         // 0=fp16, 1=fp32, 2=int32, 3=int64
        } tensor;
        struct { char *module; char *name; } ref;
    };
};

// --- Tensor record (extracted output) ---
typedef struct {
    char     name[256];
    int      data_file;       // index into data/N
    int64_t  offset;          // byte offset within data file
    int64_t  size_bytes;      // tensor size in bytes (NOT storage size)
    int      ndim;
    int64_t  shape[8];
    int      dtype;           // 0=fp16, 1=fp32, 2=int32, 3=int64
} pth_tensor_t;

// --- Model config (extracted from pickle) ---
typedef struct {
    int      n_mel;           // config[0] - 1
    int      filter_channels; // config[1]
    int      ff_dim;          // config[2]
    int      p_dim;           // config[3]
    int      ques;            // config[4] (embedding dim: 256 or 768)
    int      p_kernel;        // config[5]
    int      p_heads;         // config[6]
    int      p_layers;        // config[7]
    int      upsample_rates[8];
    int      n_upsample;
    int      upsample_initial;
    int      upsample_kernels[8];
    int      resblock_kernel;
    int      resblock_ch;
    int      sr;              // sample rate (48000, 32000, etc.)
    char     version[8];      // "v2"
    char     info[64];
    int      f0;
    int      frame_rate;      // computed: sr / prod(upsample_rates)
} pth_config_t;

// --- Parser context ---
#define PK_MAX_TENSORS 2048
#define PK_MAX_MEMO    8192
#define PK_STACK_DEPTH 16384

typedef struct {
    // Input pickle data
    const uint8_t *data;
    size_t         len;
    size_t         pos;

    // Stack
    pkv_t *stack[PK_STACK_DEPTH];
    int    sp;

    // Mark stack (for MARK/TUPLE/APPENDS/SETITEMS)
    int    marks[64];
    int    mark_sp;

    // Memo table
    pkv_t *memo[PK_MAX_MEMO];

    // Extracted tensors
    pth_tensor_t tensors[PK_MAX_TENSORS];
    int          n_tensors;

    // Extracted config
    pth_config_t config;
    int          has_config;

    // Current tensor name being built (set when we see the key)
    char         cur_tensor_name[256];
    int          in_weight_dict;

    // Error
    int          error;
    char         error_msg[256];
} pickle_ctx_t;

// Parse a pickle buffer. Returns 0 on success, -1 on error.
// On success, ctx->tensors and ctx->config are populated.
int pickle_parse(pickle_ctx_t *ctx, const uint8_t *data, size_t len);
void pk_derive_config(pickle_ctx_t *ctx);
void pk_stats(void);

// Helper: create a pkv_t value
pkv_t *pkv_new_int(int64_t v);
pkv_t *pkv_new_str(const char *s, size_t len);
pkv_t *pkv_new_bool(int v);
pkv_t *pkv_new_tuple(int count);
pkv_t *pkv_new_list(int count);
pkv_t *pkv_new_dict(int count);
pkv_t *pkv_new_ref(const char *module, const char *name);

// Free a pkv_t tree
void pkv_free(pkv_t *v);

// Dtype helpers
int  pkv_dtype_from_name(const char *name);  // "Half"→0, "Float"→1, etc.
int  pkv_dtype_element_size(int dtype);       // 0→2, 1→4, 2→4, 3→8

#endif // SPARTACVC_PICKLE_H

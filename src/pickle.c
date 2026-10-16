// pickle.c — targeted .pth pickle parser (no general-purpose VM)
//
// The .pth pickle stream has a very specific structure:
//   top_dict = {
//     "weight": { tensor_name: OrderedDict(_rebuild_tensor_v2(storage, off, shape, ...)), ... },
//     "config": [n_mel+1, filter_ch, ff_dim, p_dim, ques, p_kernel, p_heads, p_layers, ...],
//     "info": ..., "sr": ..., "f0": ..., "version": ...
//   }
//
// Each tensor is encoded as:
//   OrderedDict(
//     _rebuild_tensor_v2(
//       storage=("storage", dtype_ref, data_file, device, size),  # via BINPERSID
//       offset=0,
//       shape=(d0, d1, ...),     # TUPLE of BININTs
//       stride=(s0, s1, ...),    # TUPLE of BININTs
//       requires_grad=False,
//       backward_hooks=OrderedDict(),
//       metadata=()
//     )
//   )
//
// We walk the opcode stream and extract:
//   - tensor names (from SETITEMS keys)
//   - shapes (from TUPLE2/TUPLE3 of BININTs that follow BINPERSID)
//   - dtypes, data_file indices, sizes (from BINPERSID args)
//   - the config list values
//
// Key insight: the shape tuple is NOT inside the REDUCE args tuple.
// It sits in the vstack BETWEEN the storage value and the offset int.
// The pattern in vstack after BINPERSID is:
//   [storage, 0, shape_tuple, stride_tuple, False, OrderedDict, ()]
// The outer TUPLE (MARK) collects all of these as the args for the
// outer REDUCE (OrderedDict call). So the shape is at vstack[storage_idx+2].

#include "pickle.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int pkv_dtype_from_name(const char *name) {
    if (strcmp(name, "Half") == 0) return 0;
    if (strcmp(name, "Float") == 0) return 1;
    if (strcmp(name, "Int") == 0) return 2;
    if (strcmp(name, "Long") == 0) return 3;
    if (strcmp(name, "Byte") == 0) return 4;
    if (strcmp(name, "Bool") == 0) return 5;
    return -1;
}

int pkv_dtype_element_size(int dtype) {
    switch (dtype) {
    case 0: return 2;  // fp16
    case 1: return 4;  // fp32
    case 2: return 4;  // int32
    case 3: return 8;  // int64
    case 4: return 1;  // byte
    case 5: return 1;  // bool
    default: return 0;
    }
}

// Minimal value tag — no heap allocation
typedef enum {
    V_INT, V_STR, V_TUPLE, V_LIST, V_REF, V_STORAGE, V_BOOL, V_OTHER
} vtag_t;

typedef struct {
    vtag_t tag;
    union {
        int64_t i;
        struct { const char *data; int len; } str;  // points into pickle buffer
        struct { int count; int items[8]; } tuple;  // indices into vstack
        struct { int count; int items[32]; } list;
        struct { const char *module; int mlen; const char *name; int nlen; } ref;
        struct { char dtype[16]; char data_file[16]; int64_t size; } storage;
        int b;
    };
} val_t;

#define VSTACK_MAX 32768

static val_t vstack[VSTACK_MAX];
static int vwrite = 0;  // append-only write pointer
static int vsp = 0;     // stack top (read pointer)
static int marks[256];
static int mark_sp = 0;

static void vpush(val_t v) {
    if (vwrite >= VSTACK_MAX) vwrite = VSTACK_MAX - 1;
    vstack[vwrite++] = v;
    vsp = vwrite;
}
static val_t vpop(void) {
    if (vsp <= 0) vsp = 1;
    return vstack[--vsp];
}
static val_t vtop(void) {
    return vstack[vsp - 1];
}
static int vvalid(int idx) {
    return idx >= 0 && idx < vwrite;
}

// Memo table: maps BINPUT index → vstack position
#define MEMO_MAX 8192
static int memo[MEMO_MAX];
static void memo_set(int idx, int vpos) {
    if (idx >= 0 && idx < MEMO_MAX) memo[idx] = vpos;
}
static int memo_get(int idx) {
    if (idx >= 0 && idx < MEMO_MAX) return memo[idx];
    return -1;
}

int pickle_parse(pickle_ctx_t *ctx, const uint8_t *data, size_t len) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->data = data;
    ctx->len = len;
    ctx->pos = 0;
    vsp = 0;
    mark_sp = 0;

    // Track config list values
    int config_vals[32] = {0};
    int config_count = 0;
    int in_config = 0;

    while (ctx->pos < len) {
        int op = data[ctx->pos++];
        if (ctx->pos >= len && op != 0x2E) break;

        switch (op) {
        case 0x80: // PROTO
            ctx->pos += 1;
            break;

        case 0x2E: // STOP
            return 0;

        case 0x28: // MARK
            marks[mark_sp++] = vsp;
            break;

        case 0x29: // EMPTY_TUPLE
            vpush((val_t){.tag = V_TUPLE, .tuple = {.count = 0}});
            break;

        case 0x74: { // TUPLE
            int mark = (mark_sp > 0) ? marks[--mark_sp] : 0;
            int n = vsp - mark;
            val_t t = {.tag = V_TUPLE};
            t.tuple.count = n > 8 ? 8 : n;
            for (int i = 0; i < t.tuple.count; i++)
                t.tuple.items[i] = mark + i;
            vsp = mark;
            vpush(t);
            break;
        }

        case 0x85: { // TUPLE1
            int idx = vsp - 1;
            vsp -= 1;
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 1, .items = {idx}}};
            vpush(t);
            break;
        }
        case 0x86: { // TUPLE2
            int base = vsp - 2;
            vsp -= 2;
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 2, .items = {base, base + 1}}};
            vpush(t);
            break;
        }
        case 0x87: { // TUPLE3
            int base = vsp - 3;
            vsp -= 3;
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 3, .items = {base, base + 1, base + 2}}};
            vpush(t);
            break;
        }
        case 0x88: { // TUPLE4
            int base = vsp - 4;
            vsp -= 4;
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 4, .items = {base, base+1, base+2, base+3}}};
            vpush(t);
            break;
        }

        case 0x4A: { // BININT (signed 32-bit LE) — 'J'
            int32_t v = (int32_t)(data[ctx->pos] | (data[ctx->pos+1]<<8) |
                                  (data[ctx->pos+2]<<16) | ((uint32_t)data[ctx->pos+3]<<24));
            ctx->pos += 4;
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }
        case 0x4B: { // BININT1 (unsigned 8-bit) — 'K'
            uint8_t v = data[ctx->pos++];
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }
        case 0x4C: { // BININT4 (signed 32-bit LE) — 'L'
            int32_t v = (int32_t)(data[ctx->pos] | (data[ctx->pos+1]<<8) |
                                  (data[ctx->pos+2]<<16) | ((uint32_t)data[ctx->pos+3]<<24));
            ctx->pos += 4;
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }
        case 0x4D: { // BININT2 (unsigned 16-bit LE) — 'M'
            uint16_t v = (uint16_t)(data[ctx->pos] | (data[ctx->pos+1]<<8));
            ctx->pos += 2;
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }
        case 0x4E: { // BININT8 (signed 64-bit LE) — 'N'
            int64_t v = (int64_t)(data[ctx->pos] | (data[ctx->pos+1]<<8) |
                                  (data[ctx->pos+2]<<16) | ((uint32_t)data[ctx->pos+3]<<24) |
                                  ((uint64_t)data[ctx->pos+4]<<32) | ((uint64_t)data[ctx->pos+5]<<40) |
                                  ((uint64_t)data[ctx->pos+6]<<48) | ((uint64_t)data[ctx->pos+7]<<56));
            ctx->pos += 8;
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }

        case 0x58: { // BINUNICODE
            uint32_t slen = data[ctx->pos] | (data[ctx->pos+1]<<8) |
                            (data[ctx->pos+2]<<16) | ((uint32_t)data[ctx->pos+3]<<24);
            ctx->pos += 4;
            val_t s = {.tag = V_STR, .str = {.data = (const char *)data + ctx->pos, .len = slen}};
            ctx->pos += slen;
            vpush(s);
            break;
        }

        case 0x71: { // BINPUT (1-byte memo index)
            int idx = data[ctx->pos++];
            memo_set(idx, vsp - 1);
            break;
        }
        case 0x72: { // LONG_BINPUT (4-byte memo index)
            uint32_t idx = data[ctx->pos] | (data[ctx->pos+1]<<8) |
                           (data[ctx->pos+2]<<16) | ((uint32_t)data[ctx->pos+3]<<24);
            ctx->pos += 4;
            memo_set((int)idx, vsp - 1);
            break;
        }

        case 0x68: { // BINGET (1-byte memo index)
            int idx = data[ctx->pos++];
            int mpos = memo_get(idx);
            if (mpos >= 0 && vvalid(mpos)) {
                vpush(vstack[mpos]);
            } else {
                vpush((val_t){.tag = V_OTHER});
            }
            break;
        }

        case 0x52: { // REDUCE
            // Stack: [func, args]
            // After BINPERSID, the vstack has:
            //   [storage, offset(INT), shape(TUPLE), stride(TUPLE), False, OrderedDict, ()]
            // Then REDUCE fires (inner: _rebuild_tensor_v2)
            // We detect the pattern: look for V_STORAGE followed by INT, TUPLE, TUPLE
            val_t args = vpop();
            val_t func = vpop();

            // Find the most recent V_STORAGE in vstack
            int storage_idx = -1;
            for (int si = vwrite - 1; si >= 0 && si > vwrite - 32; si--) {
                if (vvalid(si) && vstack[si].tag == V_STORAGE) {
                    storage_idx = si;
                    break;
                }
            }

            // Pattern in vstack after BINPERSID:
            //   [storage, offset(INT), dim1, dim2, ..., TUPLE(count=N), stride1, ..., TUPLE(count=M), ...]
            // The shape dims are individual INTs followed by a TUPLE whose count == number of dims.
            // The stride dims follow the same pattern.
            if (storage_idx >= 0 && ctx->n_tensors < PK_MAX_TENSORS) {
                int off_idx = storage_idx + 1;
                if (off_idx < vwrite && vvalid(off_idx) && vstack[off_idx].tag == V_INT) {
                    // Find the shape TUPLE: scan forward from off_idx+1
                    // Pattern: INT, INT, ..., TUPLE(count=N) where N == number of preceding INTs
                    int shape_tup_idx = -1;
                    int ndim = 0;
                    for (int si2 = off_idx + 1; si2 < vwrite && si2 - off_idx < 16; si2++) {
                        if (!vvalid(si2)) break;
                        if (vstack[si2].tag == V_INT) {
                            ndim++;
                        } else if (vstack[si2].tag == V_TUPLE && vstack[si2].tuple.count == ndim && ndim >= 1) {
                            shape_tup_idx = si2;
                            break;
                        } else {
                            break; // unexpected tag
                        }
                    }
                    
                    if (shape_tup_idx >= 0 && ndim >= 1 && ndim <= 8) {
                        // Extract shape from the individual INTs (they are in order)
                        int shape_dims[8];
                        for (int i = 0; i < ndim; i++) {
                            int di = off_idx + 1 + i;
                            if (vvalid(di) && vstack[di].tag == V_INT)
                                shape_dims[i] = vstack[di].i;
                            else
                                shape_dims[i] = 0;
                        }
                        
                        // Check if we already have this (storage + offset) pair
                        int already = 0;
                        int df = atoi(vstack[storage_idx].storage.data_file);
                        int64_t off = vstack[off_idx].i;
                        for (int ti = 0; ti < ctx->n_tensors; ti++) {
                            if (ctx->tensors[ti].data_file == df &&
                                ctx->tensors[ti].offset == off) {
                                already = 1;
                                break;
                            }
                        }
                        
                        if (!already) {
                            pth_tensor_t *t = &ctx->tensors[ctx->n_tensors];
                            memset(t, 0, sizeof(*t));
                            t->data_file = df;
                            t->offset = off;
                            t->dtype = pkv_dtype_from_name(vstack[storage_idx].storage.dtype);
                            t->ndim = ndim;
                            for (int i = 0; i < ndim; i++)
                                t->shape[i] = shape_dims[i];
                            int esz = pkv_dtype_element_size(t->dtype);
                            if (esz > 0) {
                                int64_t n = 1;
                                for (int i = 0; i < ndim; i++) n *= shape_dims[i];
                                t->size_bytes = n * esz;
                            } else {
                                t->size_bytes = vstack[storage_idx].storage.size;
                            }
                            ctx->n_tensors++;
                        }
                    }
                }
            }
            vpush((val_t){.tag = V_OTHER});
            break;
        }

        case 0x51: { // BINPERSID
            // Pop the persistent ID tuple: ("storage", dtype_ref, data_file, device, size)
            val_t persid = vpop();
            val_t storage = {.tag = V_STORAGE};
            memset(&storage.storage, 0, sizeof(storage.storage));
            if (persid.tag == V_TUPLE && persid.tuple.count >= 5) {
                // [1] = dtype ref (e.g., torch.HalfStorage)
                int dtype_idx = persid.tuple.items[1];
                if (vvalid(dtype_idx) && vstack[dtype_idx].tag == V_REF) {
                    const char *n = vstack[dtype_idx].ref.name;
                    int nl = vstack[dtype_idx].ref.nlen;
                    if (nl > 7 && strncmp(n + nl - 7, "Storage", 7) == 0) {
                        int copy = nl - 7;
                        if (copy > 15) copy = 15;
                        memcpy(storage.storage.dtype, n, copy);
                        storage.storage.dtype[copy] = '\0';
                    } else {
                        int copy = nl < 15 ? nl : 15;
                        memcpy(storage.storage.dtype, n, copy);
                        storage.storage.dtype[copy] = '\0';
                    }
                }
                // [2] = data_file (string like "0")
                int df_idx = persid.tuple.items[2];
                if (vvalid(df_idx) && vstack[df_idx].tag == V_STR) {
                    int copy = vstack[df_idx].str.len < 15 ? vstack[df_idx].str.len : 15;
                    memcpy(storage.storage.data_file, vstack[df_idx].str.data, copy);
                    storage.storage.data_file[copy] = '\0';
                }
                // [4] = size (int)
                int sz_idx = persid.tuple.items[4];
                if (vvalid(sz_idx) && vstack[sz_idx].tag == V_INT) {
                    storage.storage.size = vstack[sz_idx].i;
                }
            }
            vpush(storage);
            break;
        }

        case 0x89: // NEWFALSE
            vpush((val_t){.tag = V_BOOL, .b = 0});
            break;
        case 0x8a: // NEWTRUE
            vpush((val_t){.tag = V_BOOL, .b = 1});
            break;

        case 0x5D: // EMPTY_LIST
            vpush((val_t){.tag = V_LIST, .list = {.count = 0}});
            break;

        case 0x65: { // APPENDS
            int mark = (mark_sp > 0) ? marks[--mark_sp] : 0;
            int n = vsp - mark;
            if (mark > 0) {
                val_t *lst = &vstack[mark - 1];
                if (lst->tag == V_LIST) {
                    int new_count = lst->list.count + n;
                    if (new_count > 32) new_count = 32;
                    for (int i = lst->list.count; i < new_count && i < lst->list.count + n; i++)
                        lst->list.items[i] = mark + (i - lst->list.count);
                    lst->list.count = new_count;

                    if (lst->list.count == 18) {
                        for (int i = 0; i < 18; i++) {
                            int idx = lst->list.items[i];
                            if (vvalid(idx) && vstack[idx].tag == V_INT)
                                config_vals[i] = (int)vstack[idx].i;
                            else
                                goto not_config;
                        }
                        config_count = 18;
                        in_config = 1;
                        goto config_found;
                    }
                not_config:;
                }
            }
            vsp = mark;
            break;
        }

        config_found:
        {
            pth_config_t *c = &ctx->config;
            memset(c, 0, sizeof(*c));
            c->n_mel = config_vals[0] - 1;
            c->filter_channels = config_vals[1];
            c->ff_dim = config_vals[2];
            c->p_dim = config_vals[3];
            c->ques = config_vals[4];
            c->p_kernel = config_vals[5];
            c->p_heads = config_vals[6];
            c->p_layers = config_vals[7];
            c->upsample_initial = config_vals[13];
            c->resblock_kernel = config_vals[15];
            c->resblock_ch = config_vals[16];
            c->sr = config_vals[17];
            if (c->ques == 768) {
                c->n_upsample = 4;
                c->upsample_rates[0] = 12; c->upsample_rates[1] = 10;
                c->upsample_rates[2] = 2; c->upsample_rates[3] = 2;
            } else {
                c->n_upsample = 4;
                c->upsample_rates[0] = 8; c->upsample_rates[1] = 8;
                c->upsample_rates[2] = 2; c->upsample_rates[3] = 2;
            }
            int prod = 1;
            for (int i = 0; i < c->n_upsample; i++) prod *= c->upsample_rates[i];
            c->frame_rate = (prod > 0) ? c->sr / prod : 0;
            ctx->has_config = 1;
        }
        break;

        case 0x63: { // GLOBAL
            const char *mod_start = (const char *)data + ctx->pos;
            int mlen = 0, nlen = 0;
            while (ctx->pos < len && data[ctx->pos] != '\n') { mlen++; ctx->pos++; }
            ctx->pos++;
            const char *name_start = (const char *)data + ctx->pos;
            while (ctx->pos < len && data[ctx->pos] != '\n') { nlen++; ctx->pos++; }
            ctx->pos++;

            val_t r = {.tag = V_REF, .ref = {.module = mod_start, .mlen = mlen,
                                             .name = name_start, .nlen = nlen}};
            vpush(r);
            break;
        }

        case 0x75: { // SETITEMS
            int mark = (mark_sp > 0) ? marks[--mark_sp] : 0;
            int n = vsp - mark;
            if (mark > 0 && n >= 2) {
                int tensor_idx = 0;
                for (int i = 0; i + 1 < n; i += 2) {
                    val_t *key = &vstack[mark + i];
                    if (key->tag == V_STR) {
                        int copy = key->str.len < 255 ? key->str.len : 255;
                        char keybuf[256];
                        memcpy(keybuf, key->str.data, copy);
                        keybuf[copy] = '\0';
                        if (strchr(keybuf, '.') && tensor_idx < ctx->n_tensors) {
                            strncpy(ctx->tensors[tensor_idx].name, keybuf,
                                    sizeof(ctx->tensors[tensor_idx].name) - 1);
                            tensor_idx++;
                        }
                    }
                }
            }
            vsp = mark;
            break;
        }

        case 0x7D: // EMPTY_DICT
            vpush((val_t){.tag = V_OTHER});
            break;

        default:
            ctx->error = 1;
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "unknown opcode 0x%02X at pos %zu", op, ctx->pos - 1);
            return -1;
        }
    }

    return 0;
}

// Derive config from tensor shapes (for v2 models where config is not in pickle)
static pth_tensor_t *find_tensor(pickle_ctx_t *ctx, const char *prefix) {
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name, prefix, strlen(prefix)) == 0)
            return &ctx->tensors[i];
    }
    return NULL;
}

void pk_derive_config(pickle_ctx_t *ctx) {
    if (ctx->has_config) return;
    pth_config_t *c = &ctx->config;
    memset(c, 0, sizeof(*c));

    // enc_p.emb_phone.weight: [p_dim, ques]
    pth_tensor_t *emb = find_tensor(ctx, "enc_p.emb_phone.weight");
    if (emb && emb->ndim == 2) {
        c->p_dim = (int)emb->shape[0];
        c->ques = (int)emb->shape[1];
    }

    // enc_p.encoder.attn_layers.0.conv_q.weight: [p_dim, p_dim, p_kernel]
    pth_tensor_t *convq = find_tensor(ctx, "enc_p.encoder.attn_layers.");
    if (convq) {
        // Find conv_q.weight specifically
        for (int i = 0; i < ctx->n_tensors; i++) {
            if (strstr(ctx->tensors[i].name, "conv_q.weight")) {
                convq = &ctx->tensors[i];
                break;
            }
        }
        if (convq->ndim == 3) {
            c->p_kernel = (int)convq->shape[2];
        }
    }

    // Count encoder layers
    int max_layer = 0;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strstr(ctx->tensors[i].name, "enc_p.encoder.attn_layers.")) {
            const char *p = strstr(ctx->tensors[i].name, "attn_layers.");
            if (p) {
                int layer = atoi(p + 12);
                if (layer > max_layer) max_layer = layer;
            }
        }
    }
    c->p_layers = max_layer + 1;

    // enc_p.encoder.ffn_layers.0.conv_1.weight: [ff_dim, p_dim, 1]
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strstr(ctx->tensors[i].name, "ffn_layers.") && strstr(ctx->tensors[i].name, "conv_1.weight")) {
            if (ctx->tensors[i].ndim == 3) {
                c->ff_dim = (int)ctx->tensors[i].shape[0];
            }
            break;
        }
    }

    // dec.conv_pre.weight: [filter_ch, p_dim, kernel]
    pth_tensor_t *convpre = find_tensor(ctx, "dec.conv_pre.weight");
    if (convpre && convpre->ndim == 3) {
        c->filter_channels = (int)convpre->shape[0];
    } else if (convpre && convpre->ndim == 2) {
        c->filter_channels = (int)convpre->shape[0];
    }

    // Count total upsample layers
    c->n_upsample = 0;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name, "dec.ups.", 8) == 0) {
            const char *p = ctx->tensors[i].name + 8;
            int layer = atoi(p);
            if (layer + 1 > c->n_upsample) c->n_upsample = layer + 1;
        }
    }

    // Derive upsample rates from weight_v shapes
    // dec.ups.N.weight_v: [in_ch, out_ch, up_kernel]
    // The upsample rate = up_kernel / 2 (for typical RVC models)
    // Or: rate = in_ch / out_ch (for grouped conv)
    memset(c->upsample_rates, 0, sizeof(c->upsample_rates));
    for (int layer = 0; layer < c->n_upsample; layer++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "dec.ups.%d.weight_v", layer);
        for (int i = 0; i < ctx->n_tensors; i++) {
            if (strcmp(ctx->tensors[i].name, prefix) == 0) {
                if (ctx->tensors[i].ndim == 3) {
                    int in_ch = (int)ctx->tensors[i].shape[0];
                    int out_ch = (int)ctx->tensors[i].shape[1];
                    int up_kernel = (int)ctx->tensors[i].shape[2];
                    // RVC decoder: upsample factor = kernel_size / 2
                    // (nearest-neighbor upsample by R, then conv with kernel 2R)
                    int rate = up_kernel / 2;
                    if (rate <= 0) rate = 2;
                    c->upsample_rates[layer] = rate;
                }
                break;
            }
        }
    }

    // dec.resblocks.0.convs1.0.weight_v: [resblock_ch, resblock_ch, resblock_kernel]
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strstr(ctx->tensors[i].name, "resblocks.") && strstr(ctx->tensors[i].name, "convs1.") &&
            (strstr(ctx->tensors[i].name, "weight_v") || 
             (strstr(ctx->tensors[i].name, "weight") && !strstr(ctx->tensors[i].name, "weight_g")))) {
            if (ctx->tensors[i].ndim == 3 && ctx->tensors[i].shape[1] > 1) {
                c->resblock_ch = (int)ctx->tensors[i].shape[0];
                c->resblock_kernel = (int)ctx->tensors[i].shape[2];
                break;
            }
        }
    }

    // n_mel: from flow.flows.0.pre.weight [in_ch, n_mel, 1]
    // in_ch = p_dim, n_mel = shape[1]
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name, "flow.flows.0.pre.weight", 25) == 0) {
            if (ctx->tensors[i].ndim == 3) {
                c->n_mel = (int)ctx->tensors[i].shape[1];
            }
            break;
        }
    }
    // Fallback: emb_g.weight [n_mel+1, ques]
    if (c->n_mel == 0) {
        pth_tensor_t *embg = find_tensor(ctx, "emb_g.weight");
        if (embg && embg->ndim == 2) {
            c->n_mel = (int)embg->shape[0] - 1;
        }
    }

    // p_heads: typically 2 for p_dim=192, 4 for p_dim=256
    if (c->p_dim == 192) c->p_heads = 2;
    else if (c->p_dim == 256) c->p_heads = 4;
    else c->p_heads = 2;

    // Default values
    if (c->p_dim == 0) c->p_dim = 192;
    if (c->ques == 0) c->ques = 768;
    if (c->ff_dim == 0) c->ff_dim = 1024;
    if (c->filter_channels == 0) c->filter_channels = 512;
    if (c->p_kernel == 0) c->p_kernel = 2;
    if (c->p_layers == 0) c->p_layers = 2;
    if (c->resblock_ch == 0) c->resblock_ch = 128;
    if (c->resblock_kernel == 0) c->resblock_kernel = 3;
    if (c->n_mel == 0) c->n_mel = 80;
    if (c->n_upsample == 0) {
        c->n_upsample = 4;
        c->upsample_rates[0] = 8; c->upsample_rates[1] = 8;
        c->upsample_rates[2] = 2; c->upsample_rates[3] = 2;
    }

    // Sample rate: 48000 for standard RVC
    c->sr = 48000;

    // Compute frame rate
    int prod = 1;
    for (int i = 0; i < c->n_upsample; i++) prod *= c->upsample_rates[i];
    c->frame_rate = (prod > 0) ? c->sr / prod : 0;

    c->f0 = 1;
    strcpy(c->version, "v2");
    ctx->has_config = 1;
}

void pk_stats(void) {}

void pkv_free(pkv_t *v) {
    if (!v) return;
    switch (v->type) {
    case PKV_STR:
        free(v->str.data);
        break;
    case PKV_TUPLE:
    case PKV_LIST:
        for (int i = 0; i < v->tuple.count; i++)
            pkv_free(v->tuple.items[i]);
        free(v->tuple.items);
        break;
    case PKV_DICT:
        for (int i = 0; i < v->dict.count; i++) {
            pkv_free(v->dict.keys[i]);
            pkv_free(v->dict.vals[i]);
        }
        free(v->dict.keys);
        free(v->dict.vals);
        break;
    case PKV_REF:
        free(v->ref.module);
        free(v->ref.name);
        break;
    default:
        break;
    }
    free(v);
}

void pk_debug_names(pickle_ctx_t *ctx) {
    for (int i = 0; i < 10 && i < ctx->n_tensors; i++) {
        fprintf(stderr, "  tensor[%d] name='%s' ndim=%d shape=[", i, ctx->tensors[i].name, ctx->tensors[i].ndim);
        for (int j = 0; j < ctx->tensors[i].ndim; j++)
            fprintf(stderr, "%lld%s", (long long)ctx->tensors[i].shape[j], j+1<ctx->tensors[i].ndim?",":"");
        fprintf(stderr, "] dtype=%d size=%lld\n", ctx->tensors[i].dtype, (long long)ctx->tensors[i].size_bytes);
    }
}

// pickle.c — targeted .pth pickle parser (no general-purpose VM)
//
// The .pth pickle stream has a very specific structure:
//   top_dict = {
//     "weight": { tensor_name: _rebuild_tensor_v2(storage, off, shape, ...), ... },
//     "config": [n_mel+1, filter_ch, ff_dim, p_dim, ques, p_kernel, p_heads, p_layers,
//                0, "1", [3,7,11], [[1,3,5],[1,3,5],[1,3,5]],
//                [upsample_rates], upsample_init, [upsample_kernels],
//                resblock_kernel, resblock_ch, sr],
//     "info": ..., "sr": ..., "f0": ..., "version": ...
//   }
//
// We walk the opcode stream and extract:
//   - tensor names, shapes, dtypes, data_file indices, sizes
//   - the config list values
//
// Strategy: track a simple state machine:
//   - When we see BINPERSID, the top of our "value stack" is a storage ref
//     encoded as a tuple: ("storage", dtype_ref, data_file_str, device, size)
//   - When we see REDUCE with _rebuild_tensor_v2, the args tuple contains
//     (storage, offset, shape_tuple, stride_tuple, ...)
//   - When we see SETITEMS with a string key, that's the tensor name
//
// We use a minimal value stack (just tags + indices, no heap objects).

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

// Minimal value tag — no heap allocation
typedef enum {
    V_INT, V_STR, V_TUPLE, V_LIST, V_REF, V_STORAGE, V_BOOL, V_OTHER
} vtag_t;

typedef struct {
    vtag_t tag;
    union {
        int64_t i;
        struct { const char *data; int len; } str;  // points into pickle buffer
        struct { int count; int items[8]; } tuple;  // indices into val_stack
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

    // Track current tensor name (set by SETITEMS key)
    char cur_name[256] = {0};
    // Track whether we're inside the "weight" dict
    int in_weight = 0;
    // Track config list values
    int config_vals[32] = {0};
    int config_count = 0;
    int in_config = 0;
    int config_list_base = 0; // vstack index where config list items start

    int dbg_count = 0;
    while (ctx->pos < len) {
        int op = data[ctx->pos++];
        if (ctx->pos >= len && op != 0x2E) break;
        if (dbg_count < 60) fprintf(stderr, "  [%d] op=0x%02X vsp=%d vwrite=%d\n", dbg_count, op, vsp, vwrite);
        dbg_count++;

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
            val_t a = vpop();
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 1, .items = {vsp}}};
            (void)a;
            vpush(t);
            break;
        }
        case 0x86: { // TUPLE2
            vsp -= 2;
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 2}};
            vpush(t);
            break;
        }
        case 0x87: { // TUPLE3
            vsp -= 3;
            val_t t = {.tag = V_TUPLE, .tuple = {.count = 3}};
            vpush(t);
            break;
        }

        case 0x4A: { // BININT (4-byte LE)
            int32_t v = (int32_t)(data[ctx->pos] | (data[ctx->pos+1]<<8) |
                                  (data[ctx->pos+2]<<16) | ((uint32_t)data[ctx->pos+3]<<24));
            ctx->pos += 4;
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }
        case 0x4B: { // BININT1
            int v = data[ctx->pos++];
            vpush((val_t){.tag = V_INT, .i = v});
            break;
        }
        case 0x4D: { // BININT2
            int v = data[ctx->pos] | (data[ctx->pos+1]<<8);
            ctx->pos += 2;
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
            memo_set(idx, vsp - 1); // memoize current top
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
            if (ctx->pos < 500) fprintf(stderr, "  BINGET idx=%d mpos=%d valid=%d tag=%d\n",
                idx, mpos, mpos>=0&&vvalid(mpos), mpos>=0&&vvalid(mpos)?vstack[mpos].tag:-1);
            if (mpos >= 0 && vvalid(mpos)) {
                vpush(vstack[mpos]); // push the memoized value
            } else {
                vpush((val_t){.tag = V_OTHER});
            }
            break;
        }

        case 0x52: { // REDUCE
            val_t args = vpop();
            val_t func = vpop();
            if (ctx->pos < 20000) fprintf(stderr, "  REDUCE args.tag=%d args.count=%d func.tag=%d vsp=%d vwrite=%d\n",
                args.tag, args.tag==V_TUPLE?args.tuple.count:-1, func.tag, vsp, vwrite);

            // Check if func is _rebuild_tensor_v2
            // We can't easily check the ref name here since we used V_OTHER for BINGET
            // Instead, check if args looks like a tensor args tuple
            // args should be a tuple with first element being a storage
            
            if (args.tag == V_TUPLE && args.tuple.count >= 4) {
                // Check if first element is a storage
                int storage_idx = args.tuple.items[0];
                if (vvalid(storage_idx) && vstack[storage_idx].tag == V_STORAGE) {
                    // This is a tensor!
                    val_t *storage = &vstack[storage_idx];
                    int shape_idx = args.tuple.items[2];
                    val_t *shape = vvalid(shape_idx) ? &vstack[shape_idx] : NULL;
                    
                    if (ctx->n_tensors < PK_MAX_TENSORS) {
                        pth_tensor_t *t = &ctx->tensors[ctx->n_tensors];
                        memset(t, 0, sizeof(*t));
                        // Name will be assigned later by SETITEMS
                        t->name[0] = '\0';
                        t->data_file = atoi(storage->storage.data_file);
                        t->size_bytes = storage->storage.size;
                        t->dtype = pkv_dtype_from_name(storage->storage.dtype);
                        
                        if (shape && shape->tag == V_TUPLE) {
                            t->ndim = shape->tuple.count;
                            if (t->ndim > 8) t->ndim = 8;
                            for (int i = 0; i < t->ndim; i++) {
                                int si = shape->tuple.items[i];
                                if (vvalid(si) && vstack[si].tag == V_INT)
                                    t->shape[i] = vstack[si].i;
                            }
                        }
                        ctx->n_tensors++;
                    }
                }
            }
            // Push a placeholder result
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
                    // name is like "HalfStorage" → extract "Half"
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

        case 0x5D: // EMPTY_LIST
            vpush((val_t){.tag = V_LIST, .list = {.count = 0}});
            break;

        case 0x65: { // APPENDS
            int mark = (mark_sp > 0) ? marks[--mark_sp] : 0;
            int n = vsp - mark;
            // The list is at vstack[mark-1]
            if (mark > 0) {
                val_t *lst = &vstack[mark - 1];
                if (lst->tag == V_LIST) {
                    int new_count = lst->list.count + n;
                    if (new_count > 32) new_count = 32;
                    for (int i = lst->list.count; i < new_count && i < lst->list.count + n; i++)
                        lst->list.items[i] = mark + (i - lst->list.count);
                    lst->list.count = new_count;
                    
                    // Check if this is the config list
                    // Config list has 18 items and contains integers
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
            vsp = mark; // pop items, list is now at top
            break;
        }
        
        config_found:
        {
            // Extract config
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
            // upsample_rates and kernels are lists, not ints — we'll handle them separately
            // For now, set defaults based on ques
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
            // Read "module\nname\n"
            const char *mod_start = (const char *)data + ctx->pos;
            int mlen = 0, nlen = 0;
            while (ctx->pos < len && data[ctx->pos] != '\n') { mlen++; ctx->pos++; }
            ctx->pos++; // skip \n
            const char *name_start = (const char *)data + ctx->pos;
            while (ctx->pos < len && data[ctx->pos] != '\n') { nlen++; ctx->pos++; }
            ctx->pos++; // skip \n
            
            val_t r = {.tag = V_REF, .ref = {.module = mod_start, .mlen = mlen,
                                             .name = name_start, .nlen = nlen}};
            vpush(r);
            break;
        }

        case 0x75: { // SETITEMS
            int mark = (mark_sp > 0) ? marks[--mark_sp] : 0;
            int n = vsp - mark;
            // Dict is at vstack[mark-1]
            // Key-value pairs are at vstack[mark] to vstack[mark+n-1]
            if (mark > 0 && n >= 2) {
                // First key is at vstack[mark]
                val_t *key = &vstack[mark];
                if (key->tag == V_STR) {
                    int copy = key->str.len < 255 ? key->str.len : 255;
                    memcpy(cur_name, key->str.data, copy);
                    cur_name[copy] = '\0';
                    
                    // Assign name to last tensor only if key looks like a tensor name (has a dot)
                    if (ctx->n_tensors > 0 && strchr(cur_name, '.')) {
                        pth_tensor_t *t = &ctx->tensors[ctx->n_tensors - 1];
                        if (t->name[0] == '\0') {
                            strncpy(t->name, cur_name, sizeof(t->name) - 1);
                        }
                    }
                }
            }
            vsp = mark; // pop items, dict is now at top
            break;
        }

        case 0x7D: // EMPTY_DICT
            vpush((val_t){.tag = V_OTHER});
            break;

        default:
            // Unknown opcode — try to skip it
            ctx->error = 1;
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "unknown opcode 0x%02X at pos %zu", op, ctx->pos - 1);
            return -1;
        }
    }

    return 0;
}

void pk_stats(void) {}

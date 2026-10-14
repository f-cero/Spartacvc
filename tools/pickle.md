# .pth File Format Reference (for C parser)

A `.pth` file is a **ZIP archive** (all entries `compress_type=0`, stored uncompressed)
containing a Python pickle (`data.pkl`) and raw tensor data files (`data/0` … `data/N`).

## ZIP Container

```
Local File Header (per entry):
  Offset  Size  Field
  ------  ----  -----
  0       4     Signature: 0x04034B50 ("PK\x03\x04")
  4       2     Version needed
  6       2     Flags
  8       2     Compress type (0 = stored)
  10      4     Mod time
  14      4     Mod date
  18      4     CRC-32
  22      4     Compressed size
  26      4     Uncompressed size
  30      2     Filename length (N)
  32      2     Extra field length (E)
  34      N     Filename
  34+N    E     Extra field
  34+N+E  ----- File data (uncompressed_size bytes)

End of Central Directory (EOCD) — scan backwards from EOF for "PK\x05\x06":
  Offset  Size  Field
  ------  ----  -----
  0       4     Signature: 0x06054B50
  4       2     Disk number
  6       2     CD start disk
  8       2     Entries on this disk
  10      2     Total entries
  12      4     CD size
  16      4     CD offset (→ first central directory header)
  20      N     Comment length

Central Directory Header (per entry):
  Offset  Size  Field
  ------  ----  -----
  0       4     Signature: 0x02014B50 ("PK\x01\x02")
  ...
  42      2     Filename length
  44      2     Extra field length
  46      2     Comment length
  48      4     Local header offset (→ local file header)
  52      4     Uncompressed size
  48+...  ----- Filename, extra, comment
```

**Simplification:** Since all entries are `compress_type=0`, we can just scan
local file headers sequentially from offset 0. Each header's filename tells us
the entry name; the data follows immediately after the header.

## Observed .pth Structure (Nahida)

```
nahida-jp.pth (ZIP)
├── nahida-jp_e102_s21114/data.pkl    61,476 B   ← pickle (metadata + tensor refs)
├── nahida-jp_e102_s21114/data/0      294,912 B  ← tensor data (raw fp16 bytes)
├── nahida-jp_e102_s21114/data/1          384 B
├── ...
├── nahida-jp_e102_s21114/data/456     55,808 B
└── nahida-jp_e102_s21114/version           2 B  ← b"2\n" or similar
```

457 tensor data files (data/0 … data/456).

## Pickle Protocol 2 — Opcodes Used

Only 23 opcodes appear. Byte values from the actual file:

| Byte  | Name          | Arg format                    | Stack effect |
|-------|---------------|-------------------------------|-------------|
| 0x80  | PROTO         | 1 byte (version=2)            | — |
| 0x2E  | STOP          | none                          | end of stream |
| 0x28  | MARK          | none                          | push mark |
| 0x29  | EMPTY_TUPLE   | none                          | push () |
| 0x74  | TUPLE         | none (pop to MARK)            | pop N → push (a,b,c…) |
| 0x85  | TUPLE1        | none                          | pop 1 → push (a,) |
| 0x86  | TUPLE2        | none                          | pop 2 → push (a,b) |
| 0x87  | TUPLE3        | none                          | pop 3 → push (a,b,c) |
| 0x4A  | BININT        | 4-byte LE signed int          | push int |
| 0x4B  | BININT1       | 1-byte unsigned int           | push int |
| 0x4D  | BININT2       | 2-byte LE unsigned int        | push int |
| 0x58  | BINUNICODE    | 4-byte LE len + UTF-8 bytes   | push str |
| 0x71  | BINPUT        | 2-byte LE memo index          | memo[idx] = pop |
| 0x72  | LONG_BINPUT   | 4-byte LE memo index          | memo[idx] = pop |
| 0x68  | BINGET        | 1-byte memo index             | push memo[idx] |
| 0x52  | REDUCE        | none                          | pop args+func → "call" |
| 0x51  | BINPERSID     | none                          | pop pid+persid → "load" |
| 0x89  | NEWFALSE      | none                          | push False |
| 0x5D  | EMPTY_LIST    | none                          | push [] |
| 0x65  | APPENDS       | none (pop to MARK)            | pop N → extend top list |
| 0x63  | GLOBAL        | "module\nname\n" (NL-terminated) | push callable ref |
| 0x75  | SETITEMS      | none (pop to MARK)            | pop pairs → update top dict |
| 0x7D  | EMPTY_DICT    | none                          | push {} |

## Tensor Encoding Pattern

Each tensor in the `weight` dict follows this exact pattern:

```
1. BINUNICODE <tensor_name>          → e.g. "enc_p.emb_phone.weight"
2. GLOBAL "torch._utils\n_rebuild_tensor_v2\n"
3. MARK                              → start of args
4. MARK                              → start of persistent_id
5. BINUNICODE "storage"
6. GLOBAL "torch\n<Half|Float|Int>Storage\n"   ← DTYPE
7. BINUNICODE "<data_file_index>"    ← which data/N file
8. BINUNICODE "cuda:0"               ← device (ignore)
9. BININT <size_bytes>               ← raw byte size
10. TUPLE                            → (storage, dtype, file_idx, device, size)
11. BINPERSID                        → creates storage object
12. BININT1 <storage_offset>         → usually 0
13. <shape ints> + TUPLE/TUPLE1/TUPLE2  → shape, e.g. (192, 768)
14. <stride ints> + TUPLE/TUPLE1/TUPLE2 → stride
15. NEWFALSE                         → requires_grad=False
16. BINGET 0                         → backward_hooks=OrderedDict
17. EMPTY_TUPLE                      → metadata=()
18. REDUCE                           → _rebuild_tensor_v2(...) = Tensor
```

**Dtype mapping from GLOBAL name:**
- `HalfStorage` → float16 (2 bytes/element)
- `FloatStorage` → float32 (4 bytes/element)
- `IntStorage` → int32 (4 bytes/element)
- `LongStorage` → int64 (8 bytes/element)

**Shape verification:** `prod(shape) * element_size == size_bytes`
e.g. (192, 768) × 2 bytes = 294,912 ✓

## Top-Level Structure

```python
{
    "weight": {
        "enc_p.emb_phone.weight": Tensor(shape=(192,768), dtype=fp16, data_file="0"),
        "enc_p.emb_phone.bias":   Tensor(shape=(192,),   dtype=fp16, data_file="1"),
        ...  # 457 tensors total
    },
    "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, '1',
               [3,7,11], [[1,3,5],[1,3,5],[1,3,5]], [12,10,2,2],
               512, [24,20,4,4], 109, 256, 48000],
    "info": "102epoch",
    "sr": "48k",
    "f0": 1,
    "version": "v2"
}
```

### Config array indices (RVC v2):

| Index | Value (Nahida) | Meaning |
|-------|----------------|---------|
| 0     | 1025           | n_mel + 1 (mel bins) |
| 1     | 32             | filter_channels |
| 2     | 192            | ff_dim (feedforward) |
| 3     | 192            | p_dim (encoder dim) |
| 4     | 768            | ques (embedding dim) |
| 5     | 2              | p_kernel (encoder kernel) |
| 6     | 6              | p_heads (encoder heads) |
| 7     | 3              | p_layers (encoder layers) |
| 8     | 0              | p_bihidden? |
| 9     | '1'            | p_bi? |
| 10    | [3,7,11]       | sym_upsample_rates? |
| 11    | [[1,3,5]×3]   | sym_upsample_kernel_sizes? |
| 12    | [12,10,2,2]   | upsample_rates (product=480) |
| 13    | 512            | upsample_initial_channel |
| 14    | [24,20,4,4]   | upsample_kernel_sizes |
| 15    | 109            | resblock kernel size |
| 16    | 256            | resblock channels |
| 17    | 48000          | **sr (sample rate)** |

**Frame rate = sr / prod(upsample_rates) = 48000 / (12×10×2×2) = 48000/480 = 100 fps**

## C Parser Strategy

1. **ZIP scan:** Read local file headers sequentially. Build a map:
   `filename → (offset, size)`. Identify `data.pkl` and `data/N` files.

2. **Pickle walk:** Stack-based interpreter with the 23 opcodes above.
   - Values: int, string, tuple, list, dict, bool, storage_ref, tensor
   - Memo table: indexed by BINPUT/LONG_BINPUT/BINGET
   - Intercept `BINPERSID`: extract (dtype, data_file, size) → storage_ref
   - Intercept `REDUCE` with `_rebuild_tensor_v2`: build tensor record

3. **Output:** Structured C data:
   ```c
   struct pth_tensor {
       char name[256];
       int data_file;        // index into data/N
       int64_t size_bytes;
       int ndim;
       int64_t shape[8];
       int dtype;            // 0=fp16, 1=fp32, 2=int32, 3=int64
   };
   
   struct pth_config {
       int n_mel, filter_ch, ff_dim, p_dim, ques;
       int p_kernel, p_heads, p_layers;
       int upsample_rates[8]; int n_upsample;
       int upsample_initial;
       int upsample_kernels[8];
       int resblock_kernel, resblock_ch;
       int sr;
       char version[8];
       char info[64];
   };
   ```

4. **Tensor data access:** `data_file` index → look up `data/<N>` in ZIP map
   → read `size_bytes` from that offset. Raw bytes are the tensor data
   (row-major, matching the shape).

# PyTorch `.pth` / Python Pickle Protocol 2 Reference

This reference is for parsing PyTorch `.pth` checkpoint files without relying on Python one-liners.

It covers:

1. Python pickle protocol 2 opcodes.
2. A practical 23-opcode subset commonly needed for PyTorch `.pth` files.
3. PyTorch’s tensor serialization pattern:
   - `BINPERSID` storage references
   - `torch._utils._rebuild_tensor_v2`
   - storage keys inside the ZIP container
4. ZIP container layout for `torch.save`.
5. The expected top-level checkpoint dict:

```python
{
    "weight": OrderedDict,
    "config": [...],
    "info": str,
    "sr": str,
    "f0": int,
    "version": str,
}
```

---

## 1. Basic assumptions

This document assumes:

- The pickle stream uses **protocol 2**.
- The `.pth` file is a ZIP archive, as produced by modern `torch.save`.
- ZIP entries are stored with `compress_type = 0`, i.e. uncompressed/stored.
- All fixed-size pickle integers are **little-endian**, unless explicitly noted.
- ZIP fields are little-endian.
- PyTorch raw storage files are plain binary arrays, not pickle streams.

A modern PyTorch `.pth` file usually starts with:

```text
50 4B 03 04
```

which is the ZIP local file header signature:

```text
PK\x03\x04
```

Inside the ZIP, the main pickle stream is commonly:

```text
archive/data.pkl
```

and raw tensor storage blobs are commonly:

```text
archive/data/0
archive/data/1
archive/data/2
...
```

The exact prefix may vary, but the persistent storage key in the pickle usually refers to the numeric part, e.g. `"0"`, `"1"`, etc.

---

## 2. Pickle stack-machine model

Pickle is a stack-based bytecode.

The parser maintains:

| State | Purpose |
|---|---|
| stack | Temporary object stack. |
| memo | Integer-keyed object cache. |
| mark stack | Nested markers used by container-building opcodes. |

Important concepts:

### `MARK`

`MARK` pushes a mark onto the stack.

Later opcodes such as:

- `TUPLE`
- `SETITEMS`
- `APPENDS`
- `LIST`
- `DICT`

consume everything above the most recent mark.

### Memo opcodes

- `BINPUT` stores the top stack object in memo.
- `BINGET` pushes a memoized object back onto the stack.

These are used heavily for repeated objects, especially storages, globals, and sometimes strings.

### Persistent IDs

PyTorch uses pickle persistent IDs to avoid serializing tensor storage inline.

The opcode:

```text
BINPERSID
```

pops a persistent-id object from the stack and replaces it with the result of a custom `persistent_load()` call.

In a raw parser, you should not execute arbitrary code. Instead, interpret the persistent-id tuple and map it to the correct ZIP entry.

---

## 3. The 23-opcode subset for this `.pth` pattern

The following 23 opcodes are enough for the common PyTorch checkpoint structure you described:

- top-level dict
- nested `OrderedDict`
- config list
- strings
- ints
- PyTorch tensor rebuild calls
- storage persistent IDs

If your exact file contains a close variant, see the appendix for nearby opcodes such as `TUPLE3`, `NEWTRUE`, `LONG_BINPUT`, `LONG_BINGET`, and `BINFLOAT`.

---

### 3.1 Opcode table

| # | Opcode | Byte | Arg format | Stack effect / meaning |
|---:|---|---:|---|---|
| 1 | `PROTO` | `0x80` | 1-byte unsigned int | Starts pickle stream. Argument is protocol version. Usually `0x02`. |
| 2 | `STOP` | `0x2E` | none | End of pickle stream. |
| 3 | `MARK` | `0x28` | none | Pushes a mark. Used to delimit items for tuples, lists, dicts, etc. |
| 4 | `EMPTY_DICT` | `0x7D` | none | Pushes empty dict `{}`. |
| 5 | `EMPTY_LIST` | `0x5D` | none | Pushes empty list `[]`. |
| 6 | `EMPTY_TUPLE` | `0x29` | none | Pushes empty tuple `()`. |
| 7 | `TUPLE` | `0x74` | none | Pops items after mark, pushes tuple. |
| 8 | `TUPLE1` | `0x85` | none | Pops 1 item, pushes 1-tuple. |
| 9 | `TUPLE2` | `0x86` | none | Pops 2 items, pushes 2-tuple. |
| 10 | `SETITEMS` | `0x75` | none | Pops mark and key/value pairs above it; updates dict below mark. |
| 11 | `APPENDS` | `0x65` | none | Pops mark and items above it; extends list below mark. |
| 12 | `BINUNICODE` | `0x58` | 4-byte LE length + UTF-8 bytes | Pushes Python `str`. |
| 13 | `BININT1` | `0x4B` | 1-byte unsigned int | Pushes int in range `0..255`. |
| 14 | `BININT2` | `0x4D` | 2-byte LE unsigned int | Pushes int in range `0..65535`. Common for tensor dimensions. |
| 15 | `BININT` | `0x4A` | 4-byte LE signed int | Pushes 32-bit signed int. Common for large storage sizes. |
| 16 | `BINPUT` | `0x71` | 1-byte memo index | Stores top stack object in memo. Does not pop it. |
| 17 | `BINGET` | `0x68` | 1-byte memo index | Pushes memoized object. |
| 18 | `GLOBAL` | `0x63` | Two newline-terminated strings: `module\nname\n` | Pushes named global class/function. Do not execute in a safe parser. |
| 19 | `REDUCE` | `0x52` | none | Pops args tuple and callable, pushes `callable(*args)`. In raw parsing, record the call. |
| 20 | `BUILD` | `0x62` | none | Pops state object and applies it to object below. Used for dict-like object construction. |
| 21 | `BINPERSID` | `0x51` | none | Pops persistent ID object, pushes result of `persistent_load(pid)`. For PyTorch: storage reference. |
| 22 | `NEWFALSE` | `0x89` | none | Pushes `False`. Common for `requires_grad=False`. |
| 23 | `NONE` | `0x4E` | none | Pushes `None`. Common for `backward_hooks=None`. |

---

### 3.2 Argument encoding notes

#### Fixed integers

Pickle fixed integers in protocol 2 are little-endian.

| Opcode | Size | Signedness |
|---|---:|---|
| `BININT1` | 1 byte | unsigned |
| `BININT2` | 2 bytes | unsigned |
| `BININT` | 4 bytes | signed |

There is **no fixed 8-byte integer opcode** in pickle protocol 2.

Large integers are encoded with:

- `LONG1`
- `LONG4`

These are length-prefixed arbitrary-precision two’s-complement integers.

However, in normal PyTorch `.pth` files, large sizes usually fit in `BININT`.

#### Strings

The main string opcode in protocol 2 PyTorch files is:

```text
BINUNICODE
```

Layout:

```text
58 LL LL LL LL UTF-8 bytes...
```

where:

```text
LL LL LL LL
```

is a 4-byte little-endian unsigned byte length.

The length is the number of UTF-8 bytes, not necessarily the number of characters.

#### Globals

`GLOBAL` is not length-prefixed.

Its argument is:

```text
module_name \n object_name \n
```

Example:

```text
torch._utils\n_rebuild_tensor_v2\n
```

In hex, `GLOBAL` is:

```text
63
```

followed by:

```text
module bytes
0A
name bytes
0A
```

---

## 4. PyTorch tensor serialization pattern

PyTorch tensors are not stored inline in the pickle stream.

Instead, a tensor is reconstructed by calling:

```python
torch._utils._rebuild_tensor_v2(...)
```

via pickle’s `REDUCE` opcode.

The tensor’s underlying storage is loaded through a persistent ID using:

```text
BINPERSID
```

---

## 4.1 Persistent storage ID

A PyTorch storage reference is usually encoded as a persistent-id tuple.

Common legacy/current shape:

```python
(
    "storage",
    storage_type,
    key,
    location,
    size,
)
```

For example:

```python
(
    "storage",
    torch.FloatStorage,
    "0",
    "cpu",
    1048576,
)
```

Opcode sequence:

```text
MARK
    BINUNICODE "storage"
    GLOBAL "torch" "FloatStorage"
    BINUNICODE "0"
    BINUNICODE "cpu"
    BININT 1048576
TUPLE
BINPERSID
```

After `BINPERSID`, a normal Python unpickler would call:

```python
persistent_load(pid_tuple)
```

For raw parsing, treat this as:

```text
storage_reference(key="0", storage_type=torch.FloatStorage, size=1048576)
```

The corresponding raw storage bytes are usually in:

```text
archive/data/0
```

or:

```text
data/0
```

depending on the ZIP layout.

---

### 4.1.1 Persistent ID fields

| Field | Meaning |
|---|---|
| `"storage"` | Tag identifying this as a storage persistent ID. |
| `storage_type` | Global storage class, e.g. `torch.FloatStorage`. Determines dtype. |
| `key` | String key matching a raw storage entry in the ZIP. |
| `location` | Usually `"cpu"`. |
| `size` | Usually element count for typed storages. May be byte count for some untyped-storage variants. |

Important:

- The ZIP entry’s uncompressed size is authoritative for how many raw bytes exist.
- The persistent ID size is useful for validation and dtype inference.
- For typed storages:

```text
raw_bytes = size * dtype_itemsize
```

For untyped storages, the size may already be in bytes.

---

### 4.1.2 Persistent ID variants

Depending on PyTorch version, you may see variations:

#### Common tagged form

```python
("storage", storage_type, key, location, size)
```

#### Sometimes newer/typed/untyped forms may omit the tag

```python
(storage_type, key, location, size)
```

#### Untyped storage

Some newer PyTorch versions may use:

```python
torch.UntypedStorage
```

or otherwise store raw byte counts.

When parsing defensively:

1. If first tuple element is the string `"storage"`, skip it.
2. Find the storage-type global.
3. Find the string key.
4. Find the integer size.
5. Validate against ZIP entry size if possible.

---

## 4.2 `_rebuild_tensor_v2`

The main tensor reconstruction function is:

```python
torch._utils._rebuild_tensor_v2
```

In pickle bytecode:

```text
GLOBAL "torch._utils" "_rebuild_tensor_v2"
```

Then the args tuple is built and `REDUCE` is called.

Logical Python call:

```python
torch._utils._rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
)
```

Some newer PyTorch variants may include an additional optional metadata argument:

```python
torch._utils._rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata,
)
```

---

### 4.2.1 `_rebuild_tensor_v2` argument table

| Index | Argument | Pickle representation | Meaning |
|---:|---|---|---|
| 0 | `storage` | result of `BINPERSID` or memoized storage | Underlying raw storage. Determines dtype. |
| 1 | `storage_offset` | int | Element offset into storage. Usually `0`. |
| 2 | `size` | tuple of ints | Tensor shape. |
| 3 | `stride` | tuple of ints | Tensor strides, in elements. |
| 4 | `requires_grad` | `NEWTRUE` / `NEWFALSE` | Usually `False` in state dicts. |
| 5 | `backward_hooks` | `NONE` or dict-like | Usually `None`. |
| 6 | `metadata` | optional | May be absent. Often not present in older files. |

---

### 4.2.2 Dtype is not explicit in `_rebuild_tensor_v2`

The dtype is usually inferred from the storage type in the persistent ID.

Examples:

| Storage global | Dtype | Itemsize |
|---|---|---:|
| `torch.FloatStorage` | `float32` | 4 |
| `torch.HalfStorage` | `float16` | 2 |
| `torch.BFloat16Storage` | `bfloat16` | 2 |
| `torch.DoubleStorage` | `float64` | 8 |
| `torch.LongStorage` | `int64` | 8 |
| `torch.IntStorage` | `int32` | 4 |
| `torch.ShortStorage` | `int16` | 2 |
| `torch.CharStorage` | `int8` | 1 |
| `torch.ByteStorage` | `uint8` | 1 |
| `torch.BoolStorage` | `bool` | 1 |

For a raw parser, maintain a mapping from the `GLOBAL` module/name to dtype and itemsize.

---

## 4.3 Example tensor opcode pattern

A typical 2D tensor rebuild may look like this:

```text
GLOBAL "torch._utils" "_rebuild_tensor_v2"

MARK                               # begin args tuple

    MARK                           # begin persistent ID tuple
        BINUNICODE "storage"
        GLOBAL "torch" "FloatStorage"
        BINUNICODE "0"
        BINUNICODE "cpu"
        BININT 1048576
    TUPLE                          # pid tuple
    BINPERSID                      # storage object/reference

    BININT1 0                      # storage_offset

    BININT2 1024                   # size[0]
    BININT2 1024                   # size[1]
    TUPLE2                         # size tuple

    BININT2 1024                   # stride[0]
    BININT1 1                      # stride[1]
    TUPLE2                         # stride tuple

    NEWFALSE                       # requires_grad=False
    NONE                           # backward_hooks=None

TUPLE                              # args tuple
REDUCE                             # tensor object
```

For a 1D tensor, the size and stride tuples may use `TUPLE1`:

```text
BININT2 768
TUPLE1
```

For a 3D tensor, you may see `TUPLE3`:

```text
BININT1 3
BININT1 256
BININT1 256
TUPLE3
```

For higher-rank tensors, PyTorch usually uses:

```text
MARK
    dim0
    dim1
    dim2
    dim3
TUPLE
```

---

## 4.4 Tensor byte layout inside raw storage

Given:

```text
storage_offset
size
stride
dtype.itemsize
```

the tensor views the storage as:

```text
storage_bytes[storage_offset * itemsize : ...]
```

If the tensor is contiguous, the tensor’s logical data length is:

```text
numel = product(size)
tensor_bytes = numel * itemsize
```

For non-contiguous tensors, you must honor `stride`.

The minimum number of storage elements required by a strided view is usually:

```text
1 + sum((size[i] - 1) * stride[i])
```

The persistent storage size may be larger than any one tensor view because storages can be shared.

---

## 4.5 Memoization of storages and globals

PyTorch pickle streams often memoize:

- storage objects
- global functions such as `_rebuild_tensor_v2`
- sometimes strings
- sometimes dict/list objects

You may see:

```text
BINPUT 5
```

to store an object in memo slot 5.

Later:

```text
BINGET 5
```

pushes that same object again.

For tensor parsing, track memo entries for:

- persistent storage references
- global functions
- previously built tuples or dicts if needed

If you only care about extracting raw weights, the most important memo objects are storage references.

---

## 5. ZIP container format used by `torch.save`

Modern `torch.save` produces a normal ZIP file.

The main pickle file is usually:

```text
archive/data.pkl
```

Raw tensor storage files are usually:

```text
archive/data/0
archive/data/1
archive/data/2
...
```

For stored/uncompressed entries:

```text
compression method = 0
compressed size == uncompressed size
```

So extraction is just a matter of finding offsets and sizes.

---

## 5.1 ZIP local file header

Signature:

```text
PK\x03\x04
```

Hex:

```text
50 4B 03 04
```

Layout:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | signature | `0x04034b50`, bytes `50 4B 03 04`. |
| 4 | 2 | version needed | Usually small integer. |
| 6 | 2 | general purpose bit flags | Watch bit 3: data descriptor. |
| 8 | 2 | compression method | `0 = stored`, `8 = deflate`. |
| 10 | 2 | last mod file time | DOS time. |
| 12 | 2 | last mod file date | DOS date. |
| 14 | 4 | crc-32 | Optional for parsing, useful for validation. |
| 18 | 4 | compressed size | For stored entries, equals uncompressed size. |
| 22 | 4 | uncompressed size | Same as compressed size for stored entries. |
| 26 | 2 | file name length | `n`. |
| 28 | 2 | extra field length | `m`. |
| 30 | n | file name | Usually UTF-8/ASCII. |
| 30+n | m | extra field | May be present. Must be skipped. |

File data starts at:

```text
data_offset = local_header_offset + 30 + file_name_length + extra_field_length
```

Do not assume the file data starts at `local_header_offset + 30`.

Always parse `file_name_length` and `extra_field_length`.

---

## 5.2 Central directory file header

Signature:

```text
PK\x01\x02
```

Hex:

```text
50 4B 01 02
```

Layout:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | signature | `0x02014b50`. |
| 4 | 2 | version made by | |
| 6 | 2 | version needed | |
| 8 | 2 | general purpose bit flags | |
| 10 | 2 | compression method | `0 = stored`. |
| 12 | 2 | last mod file time | |
| 14 | 2 | last mod file date | |
| 16 | 4 | crc-32 | |
| 20 | 4 | compressed size | |
| 24 | 4 | uncompressed size | |
| 28 | 2 | file name length | |
| 30 | 2 | extra field length | |
| 32 | 2 | file comment length | |
| 34 | 2 | disk number start | |
| 36 | 2 | internal file attributes | |
| 38 | 4 | external file attributes | |
| 42 | 4 | relative offset of local header | Important. |
| 46 | variable | file name | |
| ... | variable | extra field | |
| ... | variable | file comment | |

The central directory gives you:

```text
file_name
compressed_size
uncompressed_size
local_header_offset
compression_method
```

But to find the actual data, still parse the local header at `local_header_offset`.

---

## 5.3 End of central directory record, EOCD

Signature:

```text
PK\x05\x06
```

Hex:

```text
50 4B 05 06
```

Layout:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | signature | `0x06054b50`. |
| 4 | 2 | number of this disk | Usually `0`. |
| 6 | 2 | disk where central directory starts | Usually `0`. |
| 8 | 2 | number of central directory records on this disk | |
| 10 | 2 | total number of central directory records | |
| 12 | 4 | size of central directory | |
| 16 | 4 | offset of start of central directory | |
| 20 | 2 | comment length | |
| 22 | variable | comment | |

To parse a ZIP:

1. Scan backwards from end of file for EOCD signature.
2. Read central directory offset and count.
3. Iterate central directory headers.
4. For each file, use the local header offset to compute data offset.
5. Read `compressed_size` bytes.
6. If compression method is `0`, those bytes are the file content.

---

## 5.4 ZIP64 notes

Most small checkpoints fit in normal ZIP fields.

Large checkpoints may require ZIP64.

If any of these fields are saturated:

```text
0xFFFFFFFF
```

or:

```text
0xFFFF
```

then ZIP64 extensions may be used.

Important ZIP64 structures:

| Structure | Signature |
|---|---:|
| ZIP64 end of central directory record | `0x06064b50` |
| ZIP64 end of central directory locator | `0x07064b50` |
| ZIP64 extra field header ID | `0x0001` |

In ZIP64 extra fields, sizes and offsets may be 8-byte little-endian integers.

For your stated case, where all entries are stored and normal-sized, you likely only need:

- local header offset
- file name length
- extra field length
- compressed size
- uncompressed size

---

## 5.5 Data descriptor caveat

If general purpose bit flag bit 3 is set:

```text
flags & 0x0008 != 0
```

then the CRC and sizes may be zero in the local file header, and a data descriptor follows the file data.

Data descriptor signature:

```text
PK\x07\x08
```

Hex:

```text
50 4B 07 08
```

However, many `torch.save` ZIP entries are straightforward stored entries with sizes present in both local and central headers.

For robust parsing, prefer sizes from the central directory.

---

## 6. Expected top-level checkpoint structure

You described the top-level Python object as:

```python
{
    "weight": OrderedDict,
    "config": [...],
    "info": str,
    "sr": str,
    "f0": int,
    "version": str,
}
```

In pickle, this is usually represented as:

```text
EMPTY_DICT
MARK
    BINUNICODE "weight"
        ... OrderedDict value ...
    BINUNICODE "config"
        ... list value ...
    BINUNICODE "info"
        BINUNICODE "..."
    BINUNICODE "sr"
        BINUNICODE "..."
    BINUNICODE "f0"
        BININT1 / BININT2 / BININT
    BINUNICODE "version"
        BINUNICODE "..."
SETITEMS
STOP
```

The top-level dict itself may also be built with a `GLOBAL`/`REDUCE` pattern if it is not a plain dict, but for a normal dict, `EMPTY_DICT` + `SETITEMS` is common.

---

## 6.1 `weight`

The `"weight"` value is usually an `OrderedDict` mapping parameter names to tensors.

Logical shape:

```python
OrderedDict(
    [
        ("module.encoder.weight", Tensor),
        ("module.encoder.bias", Tensor),
        ...
    ]
)
```

Pickle may encode `OrderedDict` in a few ways.

Common pattern:

```text
GLOBAL "collections" "OrderedDict"
EMPTY_TUPLE
REDUCE
```

Then item state is supplied using `BUILD`, `SETITEMS`, or a state list.

For raw parsing, treat it as an order-preserving dict.

---

## 6.2 `config`

The `"config"` value is a list.

Common pickle encoding:

```text
EMPTY_LIST
MARK
    item0
    item1
    item2
    ...
APPENDS
```

If the list contains ints, expect:

- `BININT1`
- `BININT2`
- `BININT`

If it contains strings, expect:

- `BINUNICODE`

If it contains floats, you may see:

- `BINFLOAT`

`BINFLOAT` is opcode:

```text
0x47
```

and uses an 8-byte IEEE-754 double, usually big-endian.

---

## 6.3 `info`, `sr`, `version`

These are strings.

Expect:

```text
BINUNICODE
```

Example:

```text
BINUNICODE "44100"
```

Even if the string looks numeric, it is still a string in the pickle.

---

## 6.4 `f0`

This is an int.

Depending on value:

| Value range | Opcode |
|---:|---|
| `0..255` | `BININT1` |
| `256..65535` | `BININT2` |
| `-2^31..2^31-1` | `BININT` |
| larger | `LONG1` or `LONG4` |

For small checkpoint metadata, `BININT1` is common.

---

## 7. High-level pickle skeleton

A simplified view of the whole `.pth` pickle stream may look like:

```text
PROTO 2

EMPTY_DICT
MARK

    BINUNICODE "weight"

        GLOBAL "collections" "OrderedDict"
        EMPTY_TUPLE
        REDUCE

        MARK
            BINUNICODE "some.layer.weight"

                GLOBAL "torch._utils" "_rebuild_tensor_v2"
                MARK
                    MARK
                        BINUNICODE "storage"
                        GLOBAL "torch" "FloatStorage"
                        BINUNICODE "0"
                        BINUNICODE "cpu"
                        BININT 123456
                    TUPLE
                    BINPERSID

                    BININT1 0

                    BININT1 256
                    BININT1 256
                    TUPLE2

                    BININT1 256
                    BININT1 1
                    TUPLE2

                    NEWFALSE
                    NONE
                TUPLE
                REDUCE

            BINUNICODE "some.layer.bias"

                GLOBAL "torch._utils" "_rebuild_tensor_v2"
                MARK
                    MARK
                        BINUNICODE "storage"
                        GLOBAL "torch" "FloatStorage"
                        BINUNICODE "1"
                        BINUNICODE "cpu"
                        BININT 256
                    TUPLE
                    BINPERSID

                    BININT1 0

                    BININT2 256
                    TUPLE1

                    BININT1 1
                    TUPLE1

                    NEWFALSE
                    NONE
                TUPLE
                REDUCE

            ...
        SETITEMS
        BUILD

    BINUNICODE "config"
        EMPTY_LIST
        MARK
            ...
        APPENDS

    BINUNICODE "info"
        BINUNICODE "..."

    BINUNICODE "sr"
        BINUNICODE "..."

    BINUNICODE "f0"
        BININT1 0

    BINUNICODE "version"
        BINUNICODE "..."

SETITEMS

STOP
```

The exact construction of `OrderedDict` may differ slightly between Python versions, but the tensor/storage pattern is usually stable.

---

## 8. Practical parsing strategy

A raw parser for this file format can work in phases.

### Phase 1: Parse ZIP

1. Locate EOCD.
2. Parse central directory.
3. Build map:

```text
entry_name -> local_header_offset
```

4. Extract:

```text
archive/data.pkl
archive/data/*
```

or equivalent.

For stored entries, no decompression is needed.

---

### Phase 2: Parse pickle opcodes

Parse `data.pkl` sequentially.

Maintain:

```text
stack
memo
mark_stack
current_dict_key_stack
```

For most weight extraction, you do not need to implement full Python semantics.

You only need to recognize:

- dict boundaries
- `OrderedDict` boundaries
- string keys
- tensor rebuild calls
- persistent storage IDs

---

### Phase 3: Identify tensor rebuilds

When you see:

```text
GLOBAL "torch._utils" "_rebuild_tensor_v2"
```

set a state flag indicating that the next `REDUCE` args tuple describes a tensor.

Then parse the args tuple and extract:

```text
storage_reference
storage_offset
size
stride
requires_grad
backward_hooks
```

---

### Phase 4: Resolve storage references

When you see:

```text
BINPERSID
```

the top of the stack should be the persistent ID tuple.

Extract:

```text
storage_type
key
location
size
```

Map:

```text
key -> archive/data/key
```

Then read the corresponding ZIP entry.

---

### Phase 5: Associate tensors with names

The tensor name is usually the most recent dict key seen before the tensor rebuild.

For example:

```text
BINUNICODE "encoder.weight"
... tensor rebuild ...
```

So associate:

```text
"encoder.weight" -> storage key "0", offset 0, shape ..., stride ...
```

For nested dicts, maintain a key path stack.

---

## 9. Safety note

Pickle is not safe for untrusted input.

Opcodes such as:

- `GLOBAL`
- `REDUCE`
- `BUILD`
- `NEWOBJ`

can cause arbitrary code execution in a normal Python unpickler.

For a raw parser:

- Do not import modules named by `GLOBAL`.
- Do not call functions named by `GLOBAL`.
- Treat `REDUCE` as a structured record.
- Treat `GLOBAL` as a string pair `(module, name)`.
- Treat `BINPERSID` as an external storage reference.

If you must load using PyTorch, use a trusted environment and be aware of `torch.load` security implications.

---

# Appendix A: Protocol 2 opcode reference

Pickle protocol 2 accepts opcodes introduced in protocols 0, 1, and 2.

Protocol 3+ opcodes such as:

- `BINBYTES`
- `SHORT_BINBYTES`
- `SHORT_BINUNICODE`
- `FRAME`
- `STACK_GLOBAL`
- `MEMOIZE`

should not appear in a strict protocol 2 stream.

If you see them, the file was probably saved with a newer protocol.

---

## A.1 Control opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x2E` | `STOP` | none | End of stream. |
| `0x80` | `PROTO` | 1-byte protocol version | Starts protocol 2+ stream. |
| `0x28` | `MARK` | none | Push mark. |
| `0x30` | `POP` | none | Discard top stack item. |
| `0x31` | `POP_MARK` | none | Pop stack through mark. |
| `0x32` | `DUP` | none | Duplicate top stack item. |

---

## A.2 Integer opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x49` | `INT` | newline-terminated ASCII decimal | Push int. |
| `0x4A` | `BININT` | 4-byte LE signed int | Push int. |
| `0x4B` | `BININT1` | 1-byte unsigned int | Push int. |
| `0x4C` | `LONG` | newline-terminated ASCII long | Push int. |
| `0x4D` | `BININT2` | 2-byte LE unsigned int | Push int. |
| `0x8A` | `LONG1` | 1-byte length + int bytes | Push arbitrary-precision int. |
| `0x8B` | `LONG4` | 4-byte LE length + int bytes | Push arbitrary-precision int. |

`LONG1` and `LONG4` integer bytes are little-endian two’s-complement.

---

## A.3 Float opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x46` | `FLOAT` | newline-terminated ASCII float | Push float. |
| `0x47` | `BINFLOAT` | 8-byte IEEE-754 double, usually big-endian | Push float. |

Note:

There is no fixed 8-byte integer opcode in protocol 2. `BINFLOAT` is an 8-byte float, not an int.

---

## A.4 String / bytes / Unicode opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x53` | `STRING` | newline-terminated escaped string | Legacy string. |
| `0x54` | `BINSTRING` | 4-byte LE length + bytes | Legacy binary string. |
| `0x55` | `SHORT_BINSTRING` | 1-byte length + bytes | Short legacy binary string. |
| `0x56` | `UNICODE` | newline-terminated raw-unicode-escape | Push Unicode string. |
| `0x58` | `BINUNICODE` | 4-byte LE length + UTF-8 bytes | Push Unicode string. |

For protocol 2 PyTorch files, the main string opcode is:

```text
BINUNICODE
```

---

## A.5 None / bool opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x4E` | `NONE` | none | Push `None`. |
| `0x88` | `NEWTRUE` | none | Push `True`. |
| `0x89` | `NEWFALSE` | none | Push `False`. |

---

## A.6 Tuple opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x29` | `EMPTY_TUPLE` | none | Push `()`. |
| `0x74` | `TUPLE` | none | Pop items after mark, push tuple. |
| `0x85` | `TUPLE1` | none | Pop 1 item, push 1-tuple. |
| `0x86` | `TUPLE2` | none | Pop 2 items, push 2-tuple. |
| `0x87` | `TUPLE3` | none | Pop 3 items, push 3-tuple. |

---

## A.7 List opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x5D` | `EMPTY_LIST` | none | Push `[]`. |
| `0x6C` | `LIST` | none | Pop items after mark, push list. |
| `0x61` | `APPEND` | none | Append top item to list below it. |
| `0x65` | `APPENDS` | none | Extend list below mark with items above mark. |

---

## A.8 Dict opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x7D` | `EMPTY_DICT` | none | Push `{}`. |
| `0x64` | `DICT` | none | Pop items after mark, push dict. |
| `0x73` | `SETITEM` | none | Pop value and key, set item on dict below. |
| `0x75` | `SETITEMS` | none | Pop mark and key/value pairs, update dict below mark. |

---

## A.9 Memo opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x70` | `PUT` | newline-terminated ASCII decimal memo id | Store top of stack in memo. |
| `0x71` | `BINPUT` | 1-byte memo id | Store top of stack in memo. |
| `0x72` | `LONG_BINPUT` | 4-byte LE memo id | Store top of stack in memo. |
| `0x67` | `GET` | newline-terminated ASCII decimal memo id | Push memoized object. |
| `0x68` | `BINGET` | 1-byte memo id | Push memoized object. |
| `0x6A` | `LONG_BINGET` | 4-byte LE memo id | Push memoized object. |

---

## A.10 Object / global / reduce opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x63` | `GLOBAL` | `module\nname\n` | Push global object. |
| `0x52` | `REDUCE` | none | Call callable with args tuple. |
| `0x62` | `BUILD` | none | Apply state to object below. |
| `0x81` | `NEWOBJ` | none | Pop args tuple and class, push `cls.__new__(cls, *args)`. |
| `0x69` | `INST` | special legacy instance opcode | Legacy instance construction. |
| `0x6F` | `OBJ` | special legacy object opcode | Legacy object construction. |

---

## A.11 Persistent ID opcodes

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x50` | `PERSID` | newline-terminated persistent ID string | Push persistent object from string ID. |
| `0x51` | `BINPERSID` | none | Pop persistent ID object, push persistent object. |

PyTorch uses `BINPERSID`.

---

## A.12 Extension opcodes

These are protocol 2 opcodes but are uncommon in PyTorch `.pth` files.

| Byte | Opcode | Arg | Meaning |
|---:|---|---|---|
| `0x82` | `EXT1` | 1-byte extension code | Push extension object. |
| `0x83` | `EXT2` | 2-byte LE extension code | Push extension object. |
| `0x84` | `EXT4` | 4-byte LE extension code | Push extension object. |

---

# Appendix B: Common variants you may see

Depending on Python and PyTorch versions, your exact opcode set may include a few additional opcodes.

## B.1 `TUPLE3`

Byte:

```text
0x87
```

Meaning:

```text
Pop 3 items, push 3-tuple.
```

Common if tensors have rank 3 or if some metadata tuple has length 3.

---

## B.2 `NEWTRUE`

Byte:

```text
0x88
```

Meaning:

```text
Push True.
```

Common if a tensor has:

```python
requires_grad=True
```

---

## B.3 `LONG_BINPUT`

Byte:

```text
0x72
```

Arg:

```text
4-byte little-endian memo index
```

Used when memo index does not fit in one byte.

---

## B.4 `LONG_BINGET`

Byte:

```text
0x6A
```

Arg:

```text
4-byte little-endian memo index
```

Used when memo index does not fit in one byte.

---

## B.5 `BINFLOAT`

Byte:

```text
0x47
```

Arg:

```text
8-byte IEEE-754 double
```

Used if `config` contains floats.

Usually big-endian.

---

## B.6 `SETITEM`

Byte:

```text
0x73
```

Meaning:

```text
Pop value, pop key, set dict[key] = value.
```

May appear for single-item dict updates.

---

## B.7 `APPEND`

Byte:

```text
0x61
```

Meaning:

```text
Append top stack item to list below it.
```

May appear for single-item list appends.

---

## B.8 Protocol 4+ opcodes

If the second byte of the pickle stream is greater than `0x02`, the file is not strict protocol 2.

Common newer opcodes:

| Byte | Opcode | Meaning |
|---:|---|---|
| `0x8C` | `SHORT_BINUNICODE` | 1-byte length + UTF-8 string. |
| `0x95` | `FRAME` | Protocol 4 framing. |
| `0x93` | `STACK_GLOBAL` | Global from module/name strings on stack. |
| `0x94` | `MEMOIZE` | Store top stack object in memo without explicit index. |
| `0x42` | `BINBYTES` | 4-byte length + bytes. |
| `0x43` | `SHORT_BINBYTES` | 1-byte length + bytes. |

If you see these, adjust your parser accordingly.

---

# Appendix C: Quick byte cheat sheet

## Pickle start

Protocol 2 stream usually begins:

```text
80 02
```

Meaning:

```text
PROTO 2
```

---

## Common PyTorch global strings

```text
torch._utils\n_rebuild_tensor_v2\n
```

```text
torch\nFloatStorage\n
```

```text
collections\nOrderedDict\n
```

---

## ZIP signatures

| Structure | Bytes | Hex |
|---|---|---:|
| Local file header | `PK\x03\x04` | `50 4B 03 04` |
| Central directory header | `PK\x01\x02` | `50 4B 01 02` |
| EOCD | `PK\x05\x06` | `50 4B 05 06` |
| Data descriptor | `PK\x07\x08` | `50 4B 07 08` |
| ZIP64 EOCD | `PK\x06\x06` | `50 4B 06 06` |
| ZIP64 EOCD locator | `PK\x06\x07` | `50 4B 06 07` |

---

## PyTorch ZIP paths

Common:

```text
archive/data.pkl
archive/data/0
archive/data/1
...
```

Persistent ID key:

```text
"0"
```

Usually maps to:

```text
archive/data/0
```

---

## Tensor rebuild tuple summary

```python
(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
)
```

Optional newer variant:

```python
(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata,
)
```

---

## Persistent ID tuple summary

Common:

```python
(
    "storage",
    storage_type,
    key,
    location,
    size,
)
```

Variant:

```python
(
    storage_type,
    key,
    location,
    size,
)
```
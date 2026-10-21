# .pth → GGUF conversion: IN-APP, IN-MEMORY, IN C (CORRECTION)

Date: 2026-07-21
Status: **BINDING REQUIREMENT** (user-corrected plan, supersedes "write a Python converter")

## The requirement (verbatim intent)

> "Your task is to create a C function in the app which converts pth to GGUF
> for use by the GGML/CUDA runtime because DirectML isn't as universal as you
> thought it was initially, meaning my V100 which doesn't support DirectML
> needs a CUDA implementation."
>
> "The conversion function is to allow transparent loading of pth models
> directly, no temporary ONNX/GGUF models saved to disc, just convert in
> memory and then load for an optimal user experience."

## What this means

1. **A C function inside spartacvc** does the .pth → GGUF conversion.
   It is part of the app, not a dev tool.
2. **In memory.** No `.gguf` or `.onnx` temp file ever touches disk.
   `spartacvc in.wav --voice nahida-jp.pth` just works.
3. **Transparent.** The user passes the `.pth`; the app converts it in
   memory and hands the weights to the ggml graph (CUDA/Vulkan/CPU).
4. `tools/pth-to-gguf.md` is a **reference spec only** (tensor naming,
   weight-norm fusion formula, reversed-shape convention, metadata keys).
   It is NOT an instruction to write a Python converter.
5. A separate Python converter is **redundant** — rvc_python already ships
   conversion tooling (`modules/onnx/export.py`), and the app must not
   depend on any of it.
6. Why this exists: DirectML does NOT work on headless accelerator cards
   (Tesla PG500/V100 in TCC mode). Those are CUDA-only. So the GGML/CUDA
   path is the universal GPU path, and it consumes GGUF, so the app must
   be able to produce GGUF from .pth itself.

## Conversion spec (from tools/pth-to-gguf.md — the reference)

- Strip prefixes: `module.`, `model.`, `net_g.`, `synthesizer.`, `generator.`
- Fuse weight-norm: `foo.weight_g` + `foo.weight_v` → `foo.weight`
  - `W[i,...] = V[i,...] * G[i] / ||V[i,...]||` (norm over all dims except 0)
  - `scale = 0` if norm == 0
- Shape convention: GGUF `ne` = **reversed** PyTorch dims, raw bytes copied
  unchanged (no transpose):
  - Linear `[out,in]` → `[in,out]`
  - Conv1d `[out,in,K]` → `[K,in,out]`
  - ConvTranspose1d `[in,out,K]` → `[K,out,in]`
  - Embedding `[vocab,dim]` → `[dim,vocab]`
- Metadata: `general.architecture="rvc"`, `rvc.version="v1"`,
  `rvc.weight_norm="fused"`, `rvc.config.json`, `general.alignment=32`.
- Graph is **C code** (static, kokoro-style), NOT stored in the GGUF.
  The GGUF is weights + metadata only.

## Implementation plan (C, in-app)

Existing assets we build on (already in `src/`, verified):
- `pth_loader.c` — C .pth reader (ZIP + pickle + tensors), byte-identical
  to torch.load. Gives us raw named tensors + config.

New modules:
- `src/gguf_writer.c` — build a GGUF byte buffer in memory (header, KV
  metadata, tensor table, F32 blobs, 32-byte alignment). ~<300 lines.
- `src/pth2gguf.c` — THE conversion function:
  `spcv_pth2gguf(const spcv_pth_model *pth, uint8_t **buf, size_t *len)`
  - name normalization (strip prefixes)
  - weight-norm fusion (formula above)
  - shape reversal
  - emits GGUF buffer + metadata (incl. rvc.config.json from .pth config)
- `src/gguf_reader.c` — read a GGUF buffer/file into a tensor table
  (name → dims, data). Used both for in-memory buffers and for
  pre-converted `.gguf` files on disk (fast path, optional).
- `src/ggml_rvc.c` (later) — static ggml graph that looks up tensors by
  name and computes on CUDA/Vulkan/CPU.

Runtime flow:
```
--voice x.pth
  → pth_loader.c  (parse ZIP+pickles, tensors in RAM)
  → pth2gguf.c    (normalize + fuse + reverse → GGUF bytes in RAM)
  → gguf_reader.c (GGUF bytes → tensor table)
  → ggml_rvc.c    (ggml graph, weights copied to backend, compute)
--voice x.gguf
  → gguf_reader.c (direct, skip conversion)
  → ggml_rvc.c
```

## Anti-loop note

This requirement was previously established and lost to auto-compaction
caused by over-thinking. If in doubt, THIS FILE is the source of truth for
the conversion architecture. Do not propose a standalone Python converter
again.

---

## STATUS: IMPLEMENTED + VERIFIED (2026-10-20)

### What was built
- `src/pth2gguf.c` / `.h` — THE in-app converter. Uses ggml's own GGUF API
  (`gguf_init_empty`, `gguf_add_tensor`, `gguf_set_tensor_data`,
  `gguf_write_to_file`) instead of a hand-rolled writer. Does name
  normalization, weight-norm fusion, reversed-shape emission, metadata.
- `src/gguf_check.c` / `.h` — verification module. Loads a `.gguf` back
  (tensor data into a ggml context) and checks structure + (given the `.pth`)
  shapes == reversed pth shapes, fused weight-norm values, and flat-copy
  data integrity.
- CLI (in `src/main.c`):
  - `--to-gguf <out>` — convert `.pth` → `.gguf` and exit.
  - `--check-gguf <f> [--voice <v.pth>]` — verify a `.gguf` (add `--voice`
    for value cross-checks) and exit.
- `build_spcv.bat` — compiles `pth2gguf.c` + `gguf_check.c`, links
  `ggml-base.lib` (from the scratch ggml build), adds `/MD` (dynamic CRT,
  required to match ggml-base.lib's `__imp_*` imports).

### Verified (real Nahida model, through spartacvc.exe)
```
$ spartacvc --voice models/voices/nahida-jp.pth --to-gguf scratch/nahida.gguf
  converted 353 tensors (249 regular, 104 fused)
  gguf file: 114802976 bytes

$ spartacvc --check-gguf scratch/nahida.gguf --voice models/voices/nahida-jp.pth
  18/18 checks passed
```
Checks: file opens · 353 tensors · `general.architecture=rvc` ·
`rvc.weight_norm=fused` · no `.weight_g`/`.weight_v` leftovers · key tensors
exist · shapes == reversed pth shapes (emb_g, conv_pre, ups.0, in_layers.0,
conv_q) · fused value `W[0]==V[0]*G[0]/||V||` · data byte-identical.

Full inference pipeline still PASS after the changes (4.10s @ 48kHz,
max|amp|=0.8933, rms=0.1325).

### BUG: stack overflow in the gguf check/conversion paths (FIXED)
`pth_model_t` is **1.42 MB** and `rvc_model_t` is **1.42 MB** (each embeds
a `pth_model_t`). Both were declared as **stack locals** in `main`, so
`main`'s frame was ~3 MB. Calling `spcv_gguf_check` (which itself declared
a 1.42 MB `pth_model_t`) pushed the total past the 4 MB `/STACK` limit →
`0xC00000FD` (STATUS_STACK_OVERFLOW) / `0xC0000409` (stack-cookie overrun),
crashing before any output.

**Fix:** heap-allocate the big structs (`pth_model_t *m = malloc(sizeof(...))`)
in `main.c` (both the `--to-gguf` `pm` and the inference `rvc`) and in
`gguf_check.c`. `main`'s frame drops to a few hundred bytes; the 4 MB stack
is now ample. (This is the same class of bug solved before — see the
2026-10-06 session log: "allocate `pth_model_t` on the heap".)

### Gotchas learned
- `/MD` (dynamic CRT) is required when linking `ggml-base.lib` (it imports
  `__imp_ldexp` etc.). Without it: `LNK2019 unresolved __imp_ldexp`.
- `gguf_get_val_u32` **asserts** on a bad key id — always guard with
  `gguf_find_key(...) >= 0` first (the `.pth` may have no embedded config).
- The reference spec's example shapes (emb_g `[1,256]`, conv_pre K=3,
  ups.0 in=12) are **examples, not Nahida's actual shapes**. Nahida:
  emb_g `[109,256]`, conv_pre `[512,192,7]`, ups.0 `[512,256,24]` (weight_norm).
  The check derives expected shapes from the `.pth`, not from the spec.

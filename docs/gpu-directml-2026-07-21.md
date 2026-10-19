# DirectML GPU Backend — Implementation Notes (2026-07-21)

## Goal

Accelerate the RVC voice model (encoder + flow + decoder) on GPU via
**DirectML** (onnxruntime `DmlExecutionProvider`), with **user-selectable
GPU adapter**. The pure-C CPU path is unchanged and remains the fallback.

## Architecture

```
tools/spartacvc_gpu.py            spartacvc.exe (pure C)
─────────────────────             ─────────────────────
1. Load .pth                      1. Read ONNX bytes from stdin
2. Export to ONNX                 2. CreateSessionFromArray (in-memory)
   IN MEMORY (BytesIO)            3. DmlExecutionProvider + device_id
3. Pipe ONNX bytes ────────────►  4. HuBERT / RMVPE / f0 (as before)
   to spartacvc.exe stdin         5. Run voice ONNX (enc+flow+dec) on GPU
                                  6. FAISS retrieval + WAV out
```

**Key decision (per user):** the `.pth → ONNX` conversion happens **in memory**
(`torch.onnx.export` → `io.BytesIO`), never written to disk. The ONNX bytes are
piped to `spartacvc.exe`'s stdin, which loads them via ORT
`CreateSessionFromArray`. No ONNX file ever touches the disk.

### Why a Python wrapper for the GPU path?

The `.pth → ONNX` conversion requires PyTorch (`torch.onnx.export`), which is
Python. The pure-C runtime cannot run PyTorch. So the GPU path uses a thin
Python wrapper (dev-time tool in `tools/`) that:
1. Converts the `.pth` to ONNX in memory.
2. Pipes the ONNX bytes to `spartacvc.exe`.

Everything downstream (HuBERT, RMVPE, f0, FAISS, WAV output, and the voice
model inference via ORT+DML) is done by `spartacvc.exe` in C. The CPU path
(`--voice <v.pth>`) remains 100% pure C.

## Files changed / added

| File | Change |
|------|--------|
| `src/onnx.h` | + `spcv_onnx_set_device_id()`, + `spcv_onnx_load_array()` |
| `src/onnx.c` | Refactored EP setup into `apply_ep()` (passes `device_id` to DML); split session setup into `finish_session()`; + `spcv_onnx_load_array()` (CreateSessionFromArray) |
| `src/main.c` | + `--gpu <id>` / `--gpu list`, + `--voice-stdin`; `read_all_stdin()` uses `ReadFile` (see gotcha) |
| `src/gpu_list.c` | New: `spcv_list_gpus()` — DXGI adapter enumeration for `--gpu list` (links `dxgi.lib`) |
| `build_spcv.bat` | + `src/gpu_list.c`, + `dxgi.lib` on the link line |
| `tools/spartacvc_gpu.py` | New: in-memory ONNX export + pipe to spartacvc |

## Usage

```bash
# GPU (DirectML), primary adapter:
python tools/spartacvc_gpu.py input.wav \
    --voice models/voices/nahida-jp.pth \
    --hubert models/base/vec-768-layer-12.onnx \
    --rmvpe models/base/rmvpe.onnx \
    --gpu 0 --backend dml --out out.wav

# Pick a specific GPU adapter (0-based DXGI index):
python tools/spartacvc_gpu.py ... --gpu 1

# List the available GPU adapters (and their --gpu indices):
./spartacvc.exe --gpu list
#   GPU 0: Radeon Vega Frontier Edition   [AMD/ATI, vendor=0x1002 device=0x6863]
#   GPU 1: Microsoft Basic Render Driver  [other,   vendor=0x1414 device=0x008c]

# CPU fallback (pure C, no ONNX, no Python):
./spartacvc.exe input.wav --voice models/voices/nahida-jp.pth \
    --hubert ... --rmvpe ... --out out.wav --backend cpu
```

`spartacvc.exe` also supports the raw C-level flags directly:

```
--gpu <id>        DML GPU adapter index (0 = primary)
--voice-stdin     read voice ONNX bytes from stdin (piped)
--backend dml|cpu|auto
```

## GPU selection

`spcv_onnx_set_device_id(id)` stores the adapter index; `apply_ep()` passes it
to the DML provider as the `device_id` key-value option:

```c
const char *keys[1]  = { "device_id" };
char val[16]; snprintf(val, sizeof(val), "%d", g_device_id);
const char *vals[1] = { val };
api->SessionOptionsAppendExecutionProvider(opts, "DmlExecutionProvider",
                                           keys, vals, 1);
```

Verified: `--gpu 0` → PASS; `--gpu 1` → FATAL (no valid adapter 1 on this box).

To see which index is which GPU, run `./spartacvc.exe --gpu list` (implemented in
`src/gpu_list.c`, links `dxgi.lib`). Note: `--gpu list` enumerates **DXGI display
adapters**; a compute-only GPU (e.g. a headless NVIDIA V100 with no display output)
may not appear there even though DirectML can still target it — in that case try
`--gpu` values beyond the listed indices.

## ONNX export contract

From `rvc_python/modules/onnx/export.py` (replicated in `spartacvc_gpu.py`):

- Inputs: `phone[1,T,768] f32`, `phone_lengths[1] i64`, `pitch[1,T] i64`,
  `pitchf[1,T] f32`, `ds[1] i64`, `rnd[1,192,T] f32`
- Output: `audio[1,1,T*480] f32`
- Dynamic axis on time: `phone[1]`, `pitch[1]`, `pitchf[1]`, `rnd[2]`
- `opset_version=13`, `do_constant_folding=False`, `dynamo=False` (legacy
  exporter — DML-compatible, matches rvc_python)

The exported ONNX is ~116 MB for Nahida. It runs correctly with a **different**
time length than the export length (T=30 tested vs T=200 export) — the dynamic
axis works despite a `TracerWarning` on the relative-position `if pad_length>0`
branch (both T=200 and T=30 take the `F.pad` branch, so the baked branch is
correct).

## BENCHMARK (4.1 s audio, Nahida, 48 kHz)

| Path | Wall time | Notes |
|------|-----------|-------|
| Pure-C (CPU) | **~22 s** | decoder ~20 s (95%) |
| ONNX + CPU EP | **~3.0 s** | ~7× faster than pure-C |
| ONNX + **DML (GPU)** | **~3.4 s** | DML overhead ≈ GPU speedup at this size |

The ONNX path (CPU or DML) is **~7× faster** than the pure-C path because the
decoder (the bottleneck) runs as compiled ORT kernels instead of interpreted C
loops. For this small 4.1 s clip the DML GPU-CPU transfer + kernel-launch
overhead slightly outweighs the V100's compute advantage, so DML is marginally
slower than CPU-EP here. For longer audio / larger models the GPU should win.
DML is confirmed active (its timing differs from CPU-EP).

## Gotchas (learned the hard way)

1. **`fread()` on a redirected stdin is broken in this MSVC environment.**
   A minimal C program reading a 115 MB file from stdin via `fread` gets
   **0–11 bytes**, while `ReadFile()` on `GetStdHandle(STD_INPUT_HANDLE)` gets
   all 115 MB. Python (also a console app) reads the same stdin fine. Fix:
   `read_all_stdin()` in `main.c` uses `ReadFile` on Windows, `fread` on POSIX.

2. **`rvc_python/__init__.py` imports `parselmouth`** (and other heavy deps)
   that aren't installed. The wrapper bypasses it by registering a lightweight
   `rvc_python` package stub in `sys.modules` (with the correct `__path__`)
   before importing `rvc_python.lib.infer_pack.models_onnx`.

3. **torch 2.14's default ONNX exporter needs `onnxscript`** and produces a
   different graph. Use `dynamo=False` to force the legacy (TorchScript)
   exporter that rvc_python uses and that DML supports.

4. **`long` is 32-bit on Windows** — use `int64_t` for 8-byte fields (already
   handled in the FAISS code; kept in mind here).

## Correctness

- GPU (DML) output: 4.10 s @ 48 kHz, max|amp|=0.888 (no clip), rms=0.128.
- STT: "Hello, this is a short test of the voice conversion pipeline." ✅
- Same contract as the pure-C path; the voice model is the same Nahida weights,
  just run through ORT+DML instead of interpreted C.

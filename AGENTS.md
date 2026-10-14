# SPARTACVC — AGENT NOTES (READ THIS EVERY SESSION)

## NON-NEGOTIABLE RULES

1. **This is a PURE C project.** No Python at runtime. Ever. `spartacvc.py` is a
   reference oracle for validating C output — it is NOT the product. The product
   is a C binary. If a Python file appears in the repo it is `tools/reference_*.py`
   only, clearly marked as a dev-time validation tool.

2. **When blocked, ASK THE USER.** Do not silently switch languages, do not
   "defer" the C port, do not paper over a bug with a different implementation.
   The user is better at fixing things. Say what is broken, what you tried,
   and ask.

3. **Heavy documentation.** Every function gets a doc comment: purpose,
   parameters, return value, edge cases, complexity if non-trivial.
   Every module gets a header comment explaining its role in the pipeline.

4. **Tracing / debug logging.** Two gates:
   - Compile-time: `#ifdef SPCV_DEBUG` (set via `-DSPCV_DEBUG` or CMake `DEBUG=ON`)
   - Runtime: `--debug` CLI flag or `SPCV_DEBUG=1` env var
   - BOTH must be true for traces to emit. Traces go to `spcv_trace.log`
     (append mode) AND stderr. Format: `[spcv] <module> <msg>`.
   - In release builds (`SPCV_DEBUG` not defined), all trace calls compile to
     nothing. Zero overhead.

5. **Structure first, code second.** Plan the full module layout, data
   structures, and function signatures BEFORE writing implementations.
   Fill in the scaffolding. Do not ad-hoc.

6. **Never search outside the project directory** unless explicitly told to.

7. **Git identity:** `f-cero` / `5697283+f-cero@users.noreply.github.com`

## PIPELINE (the thing we are building in C)

```
input.wav
  |
  v
[io]         read WAV, resample to 16 kHz          (lib: miniaudio or raw RIFF)
  |
  v
[frontend]   HuBERT ONNX inference -> [1,T,256|768] (lib: onnxruntime C API)
  |          resample frames to model fps
  v
[frontend]   mel spectrogram (128 bins, 1024 fft, 160 hop)  (lib: custom DFT)
  |
  v
[f0]         RMVPE ONNX inference -> f0 [Hz] per frame      (lib: onnxruntime C API)
  |          cents decode, 9-bin averaging
  |          pitch shift: f0 *= 2^(semitones/12)
  v
[backend]    prepare pitch inputs (log-f0, quantized)
  |
  v
[backend]    Voice decoder ONNX inference -> raw PCM       (lib: onnxruntime C API)
  |
  v
[io]         normalize, write WAV                         (lib: miniaudio or raw RIFF)
output.wav
```

## MODULE LAYOUT (planned)

```
src/
  spartacvc.h       — public API, version, build flags
  spartacvc.c       — main(), arg parsing, pipeline orchestration
  debug.h           — SPCV_TRACE(), SPCV_LOG(), debug gate macros
  io.h / io.c       — WAV read/write, resampling
  dft.h / dft.c     — FFT, mel spectrogram
  hubert.h / hubert.c — HuBERT session wrapper
  f0.h / f0.c       — RMVPE session + cents decode + pitch shift
  voice.h / voice.c — voice decoder session wrapper
  onnx.h / onnx.c   — ONNX Runtime C API wrapper (env, session, run)
  manifest.h / manifest.c — .spacvc manifest reader (done)
  wav.h / wav.c     — RIFF/WAV parser + writer (no external dep)
tools/
  pth2manifest.py   — .pth -> .spacvc converter (dev-time only)
  reference_pipeline.py — Python oracle for validation (dev-time only)
```

## BUILD

- MSVC on Windows (VS 2022 Professional)
- CMake for cross-platform
- ONNX Runtime: static link or DLL, version TBD (1.20.1 has a bug)
- No other external deps (WAV I/O is hand-rolled, FFT is hand-rolled)

## RESOLVED: ORT C API "bad allocation" (2026-10-14)

Root cause: FOUR bugs in the C ORT wrapper, found via rubber duck debug:
1. `Run()` param order: `output_names, output_names_len, outputs` (NOT `outputs, len`)
2. `CastTypeInfoToTensorInfo` returns a VIEW into `OrtTypeInfo` — do NOT
   call `ReleaseTensorTypeAndShapeInfo(tsi)` separately. Only `ReleaseTypeInfo(ti)`.
3. WAV RIFF reader: chunk header is 4-byte name + 4-byte size. Was reading
   8 bytes then 4 more (double-read).
4. ONNX elem type 1 = FLOAT (4 bytes), NOT FLOAT16 (2 bytes).

**Lesson: when the ORT C API does something weird, check the header docs
for ownership semantics and exact param order. Don't guess.**

## ORT C API GOTCHAS (reference)

- `Run()`: `(session, run_opts, in_names, in_values, in_len, out_names, OUT_LEN, out_values)`
- `CastTypeInfoToTensorInfo(ti, &tsi)`: tsi is a VIEW, free via `ReleaseTypeInfo(ti)` only
- `SessionGetInputName` / `SessionGetOutputName`: string allocated by ORT allocator,
  free via `alloc->Free(alloc, ptr)`
- `CreateTensorWithDataAsOrtValue`: byte_size must match dims * elem_size exactly
- ONNX types: 1=FLOAT(4B), 2=UINT8(1B), 6=INT32(4B), 7=INT64(8B), 10=FLOAT16(2B)

## STYLE

- C99, no C++ constructs
- `snake_case` for functions/variables, `UPPER_SNAKE` for macros/constants
- `spcv_` prefix for all public symbols
- Every `.h` has `#ifndef` guards
- Doc comments: `/** ... */` style
- Max line length: 100 chars
- No global mutable state (pass context structs)

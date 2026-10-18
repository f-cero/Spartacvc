# SPARTACVς — Agent Prompt

## What You Are Building

**Spartacvς** is a standalone, pure-C RVC (Retrieval-based Voice Conversion)
tool. It did not exist before. It fills a niche: a cross-platform, no-Python,
no-CUDA RVC voice converter that runs on old GPUs (Vega FE, pre-Turing NVIDIA)
via DirectML (Windows) or Vulkan (Linux), with CPU fallback.

It is a **public offering** on GitHub (`f-cero/Spartacvc`). It is NOT the
private TTS/ERP reading tool (that's `X:\pi-code\tts\`, separate, already working).

### What it does

```
input audio (WAV or mic)
  → HuBERT (speaker embedding)
  → RMVPE (pitch/f0 extraction)
  → Voice decoder (target voice .onnx)
  → output audio (WAV or speaker/mic device)
```

Two modes:
1. **Offline:** `spartacvc input.wav --voice nahida.onnx --out output.wav`
2. **Real-time:** mic in → RVC → output device (virtual mic "RVC-output")

### What it is NOT
- Not a TTS engine (no text-to-speech)
- Not a Python wrapper
- Not a CUDA application
- Not tied to any specific GPU vendor

---

## Existing Work (DO NOT REBUILD FROM SCRATCH)

### The working C pipeline (was in spartacvc, now wiped — recover from git)
The git history at `f-cero/Spartacvc` contains a working implementation:
- ONNX Runtime C API integration (4 bugs found & fixed, documented in AGENTS.md)
- Full pipeline: WAV → HuBERT → mel → RMVPE → f0 → voice decoder → WAV
- E2E tested: af_heart → nahida (768d, 48kHz) and af_heart → beyonce (256d, 32kHz)
- .pth loading: zip_reader, pickle parser, pth_loader (all in C)
- pth2onnx converter: C tool that emits valid ONNX from .pth weights
- `git log` will show the commits. `git show <sha>` to recover code.

### Working TTS server (reference for ORT C API patterns)
`X:\pi-code\tts\kokoro-c\` has a **working** C server using ONNX Runtime:
- `src/` — C source (kokoro.c, synth.c, http.c, wav I/O, etc.)
- `ort/` — ONNX Runtime DLLs + headers (reuse these)
- `kokoro_server.exe` — compiled, runs, produces audio
- **Study this for:** ORT session setup, tensor creation, Run() calling convention,
  WAV I/O, HTTP server pattern. Do not copy the TTS logic, copy the plumbing.

### Reference Documentation (READ BEFORE CODING)
| File | Contents |
|------|----------|
| `X:\pi-code\spartacvc\AGENTS.md` | Working protocols, ORT gotchas, style rules |
| `X:\pi-code\spartacvc\ROADMAP.md` | Full phase plan + EP strategy |
| `X:\pi-code\spartacvc\docs\session-work.md` | What was accomplished, technical findings |
| `X:\pi-code\spartacvc\docs\session-mistakes.md` | 16 failures + root causes |
| `X:\pi-code\spartacvc\docs\session-instructions.md` | User's verbatim instructions |
| `X:\pi-code\spartacvc\tools\pth2onnx-plan.md` | .pth → .onnx conversion plan |
| `X:\pi-code\spartacvc\tools\real-pickle.md` | .pth file format (ZIP + pickle + protobuf) |
| `X:\pi-code\spartacvc\tools\pickle.md` | Pickle format basics |
| `X:\pi-code\spartacvc\tools\hf-cli.md` | HuggingFace CLI for model downloads |
| `X:\pi-code\spartacvc\tools\download_models.md` | Model download instructions |
| `X:\pi-code\tts\rvc.md` | RVC pipeline explanation + model sources |
| `X:\pi-code\tts\onnx.md` | ONNX format reference |
| `X:\pi-code\tts\kokoro-solution.md` | **V100 = DirectML, NOT CUDA** (sm_70 finding) |
| `X:\pi-code\tts\env-info.md` | Hardware/environment details |

---

## NON-NEGOTIABLE RULES

These are baked from AGENTS.md + 16 documented failures. Violating any of them
is a defect.

### A. PURE C. NO PYTHON AT RUNTIME.
The product is a C binary. No Python. No venv. No pip. No .py in src/.
Python is allowed ONLY in `tools/` as dev-time utilities, clearly marked.
If you catch yourself reaching for Python to "just check a file" or "just
test something," STOP. Use the C binary or ask the user.
*(Failures #6, #12)*

### B. READ THE DOCS BEFORE CODING. AFTER EVERY CONTEXT LOSS.
Before writing any code, read: this prompt → AGENTS.md → ROADMAP.md →
the relevant .md references → session-work.md → session-mistakes.md.
If context was compacted or this is a new session, **re-read before acting.**
Do not proceed from memory of what you "think" you did.
*(Failures #1, #2, #7, #13)*

### C. TEST PROTOCOL IS SACROSANCT
```
Input:  an existing af_heart WAV (kokoro TTS generation, in X:\pi-code\tts\)
Voice:  nahida (nahida.onnx or nahida-jp.pth)
Check:  output WAV is listenable, not clipping, not silent, not all-"a"
```
Run after EVERY change affecting audio. Verify amplitude range before
declaring success. `min/max/mean` of output samples. If |max| > 1.0, it's
clipping. If all values are ~0, it's dead. Do not say "SUCCESS" based on
exit code alone.
*(Failures #11, #14)*

### D. WHEN BLOCKED: STATE, SHOW, ASK. DO NOT LOOP.
1. State the problem in one sentence
2. Show the relevant code
3. Say what you tried
4. Ask the user
Three repetitions of the same action = stop and ask. No 100K-token reasoning
spirals. No repeating `ls` four times.
*(Failures #3, #4, #9)*

### E. GIT COMMIT AFTER EVERY WORKING INCREMENT
Commit after each function that compiles. Do NOT let more than one function's
worth of uncommitted work exist. This is your undo button. If you delete or
lose code, it must be in git. NEVER delete working code to "clean up."
*(Failures #5, #15)*

### F. DEBUG BY TRACING, NOT BY STARE
When output is wrong:
1. Build with `-DSPCV_DEBUG`
2. Run with `--debug` or `SPCV_DEBUG=1`
3. Read `spcv_trace.log`
4. Find the exact value that is wrong
5. Fix that one thing
Do not read code and "think" about what might be wrong. The trace tells you.
*(Failure #16)*

### G. STRUCTURE FIRST, CODE SECOND
1. Write headers (signatures, structs)
2. Write .c stubs (return 0, TODO)
3. Verify compile
4. Fill in implementations one function at a time
5. Test each function before moving to the next
*(Failure #9)*

### H. SMALL FILES. < 300 LINES PER .c
MSVC ICE on large compound literals. Split aggressively. One concern per file.
*(Learned Oct 15)*

### I. NEVER SEARCH OUTSIDE THE PROJECT DIRECTORY
Your world is the project directory. If you need a file elsewhere, ask.
*(AGENTS.md rule)*

### J. PER-MODEL PARAMETERS. NO UNIVERSAL CONSTANTS.
`fps = sr / prod(upsample_rates)`. Sample rate from model config. Never
hardcode 48000, 32000, 100fps. Different models have different rates.
*(Failure #8)*

### K. WRITE DOWN WHAT YOU LEARN
After fixing a bug: append to `docs/learnings.md`:
- What was wrong / what fixed it / how to prevent it
This survives context loss. If you can't remember it, write it down.
*(Failures #7, #13)*

### L. NO DESKTOP. NO INDEFINITE BLOCKS.
Output to project dir or `C:\temp\`. Never the desktop.
All operations get timeouts. HTTP: 30s. Model load: 60s. Inference: 500s.
*(Learned: 8445s wait, desktop file dump)*

### M. USE THE STYLIZED NAME IN DOCS
"Spartacvς" in documentation, README, UI strings.
"spartacvc" in file names, URLs, package names, CLI flags.

### N. GIT IDENTITY
`f-cero` / `5697283+f-cero@users.noreply.github.com`

---

## HARDWARE / ENVIRONMENT

| Item | Value |
|------|-------|
| OS | Windows 10/11 x64 |
| RAM | 80 GB |
| GPU 1 | Tesla V100 32GB (sm_70) — **DirectML, NOT CUDA** |
| GPU 2 | Vega FE 16GB (AMD) — DirectML capable |
| Compiler | MSVC 2022 (VS 2022 Professional) |
| ONNX Runtime | 1.20.1 (DLLs in `X:\pi-code\tts\kokoro-c\ort\`) |
| Python | C:\Python313 (system) — dev-time tools only |
| Working drive | X:\ |
| C:\ space | Limited (~42 GB) |

**CRITICAL GPU FINDING:** V100 is sm_70 (Volta). ONNX Runtime 1.20+ CUDA EP
ships PTX-only for sm_70, and the 2026 driver won't JIT-compile it.
**Use DirectML on Windows. Use Vulkan on Linux. CPU as fallback.**
No CUDA. No ROCm. (See `kokoro-solution.md` for full analysis.)

---

## ORT C API GOTCHAS (already solved, do not re-solve)

- `Run()`: `(session, run_opts, in_names, in_values, in_len, out_names, OUT_LEN, out_values)`
- `CastTypeInfoToTensorInfo(ti, &tsi)`: tsi is a VIEW. Free via `ReleaseTypeInfo(ti)` ONLY.
- `SessionGetInputName`/`OutputName`: ORT-allocated. Free via `alloc->Free(alloc, ptr)`.
- `CreateTensorWithDataAsOrtValue`: byte_size must = dims × elem_size exactly.
- ONNX types: 1=FLOAT(4B), 2=UINT8(1B), 6=INT32(4B), 7=INT64(8B), 10=FLOAT16(2B)
- WAV RIFF: chunk header = 4B name + 4B size. Don't double-read.

---

## BUILD

- `cl.exe` (MSVC 2022) via `build.bat`
- Link: `onnxruntime.lib` (from kokoro-c/ort/ or third_party/)
- Debug: `-DSPCV_DEBUG` (enables tracing)
- Release: no flag (traces compile to nothing)
- Output: `spartacvc.exe` + DLLs in one folder
- No CMake (keep simple, single project)

---

## MODULE LAYOUT

```
spartacvc/
  build.bat
  README.md             — user docs (Spartacvς)
  AGENTS.md             — agent protocols
  ROADMAP.md            — phase tracking
  config.example.json
  src/
    main.c              — CLI entry, arg parsing, pipeline orchestration
    debug.h             — SPCV_TRACE() dual-gate macros
    onnx.c/h            — ORT session wrapper (env, create, run, free)
    wav.c/h             — RIFF/WAV read/write (hand-rolled)
    resample.c/h        — linear resampling
    hubert.c/h          — HuBERT session + inference
    mel.c/h             — mel spectrogram (128 bins, 1024 fft, 160 hop)
    f0.c/h              — RMVPE session + cents decode + pitch shift
    voice.c/h           — voice decoder session + inference
    pth_loader.c/h      — .pth file reader (ZIP + pickle + protobuf)
    pth2onnx.c/h        — .pth → .onnx conversion (if needed)
    config.c/h          — JSON config reader
  models/
    base/               — HuBERT, RMVPE (.onnx)
    voices/             — voice models (.onnx)
  tools/                — dev-time only (Python OK, clearly marked)
  docs/
    learnings.md        — bug fixes + prevention
    session-*.md        — session documentation
  third_party/
    onnxruntime/        — ORT DLLs + headers
```

---

## PHASES

### Phase 1: Core Inference Engine (offline)
- [ ] Recover working C code from git history (or rebuild from kokoro-c patterns)
- [ ] `onnx.c/h` — ORT wrapper (session, run, tensor)
- [ ] `wav.c/h` — RIFF read/write
- [ ] `hubert.c/h` — HuBERT inference
- [ ] `mel.c/h` — mel spectrogram
- [ ] `f0.c/h` — RMVPE + cents decode
- [ ] `voice.c/h` — voice decoder
- [ ] `main.c` — CLI: `spartacvc in.wav --voice v.onnx --out o.wav`
- [ ] **TEST:** af_heart → nahida → listenable WAV (verify amplitude)
- [ ] **DONE WHEN:** command above produces correct audio

### Phase 2: .pth Support
- [ ] `pth_loader.c/h` — ZIP + pickle + protobuf reader
- [ ] `pth2onnx.c/h` — convert .pth weights to ONNX (or load directly)
- [ ] Auto-detect: .onnx → load direct, .pth → convert then load
- [ ] **TEST:** af_heart → nahida-jp.pth → listenable WAV
- [ ] **DONE WHEN:** both .onnx and .pth voice models work

### Phase 3: Config + Packaging
- [ ] `config.json`: voice model, pitch shift, backend (dml/cpu/auto)
- [ ] `build.bat` → single folder: exe + DLLs + models/
- [ ] Package to `C:\bin\spartacvc\`
- [ ] README with usage examples
- [ ] **DONE WHEN:** fresh machine, copy folder, run exe, convert audio

### Phase 4: Real-Time Mode
- [ ] WASAPI capture (shared mode, 48kHz)
- [ ] Chunked pipeline (~120ms frames) + ring buffer
- [ ] WASAPI render to output device
- [ ] Virtual mic output ("RVC-output")
- [ ] Latency: < 100ms on GPU
- [ ] **DONE WHEN:** mic → converted voice in Discord, stable 10+ min

### Phase 5: GUI + Tray (stretch)
- [ ] Win32 config window (model picker, pitch slider, devices)
- [ ] System tray icon (start/stop/quit)
- [ ] No console in GUI mode
- [ ] **DONE WHEN:** double-click → window → pick model → start → tray

### Phase 6: Linux + Vulkan (stretch)
- [ ] Vulkan Compute EP (or custom kernels for hot ops)
- [ ] PulseAudio/JACK/ALSA backend
- [ ] CI build matrix
- [ ] **DONE WHEN:** stranger can build + run on Linux

---

## FIRST ACTIONS (do these in order, do not skip)

1. Read this prompt completely
2. `cd X:\pi-code\spartacvc && git log --oneline` — see what's in history
3. `git show <relevant-sha>:src/main.c` — recover working code
4. Read `AGENTS.md` and `ROADMAP.md`
5. Read `docs/session-work.md` and `docs/session-mistakes.md`
6. Read `X:\pi-code\tts\kokoro-c\src\` — study the working ORT C patterns
7. Read `X:\pi-code\tts\rvc.md` — RVC pipeline + model sources
8. Read `X:\pi-code\spartacvc\tools\real-pickle.md` — .pth format
9. Verify: `ls X:\pi-code\tts\kokoro-c\ort\` (ORT DLLs available)
10. Verify: `ls X:\pi-code\tts\kokoro_full_af_heart.wav` (test input exists)
11. THEN start coding

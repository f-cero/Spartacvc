# Work Performed — Session Summary

Session: 2026-10-06T23:17:52Z → 2026-10-17T01:17:48Z

---

## Phase 1: TTS Pipeline (Oct 6–13)

### Oct 6–7: Koboldcpp TTS Setup
- Explored `X:\ImageIndexer\resources` koboldcpp installation
- Identified TTS model options (OuteTTS, Kokoro)
- Configured `tts-test.kcpps` settings for Vega FE
- Built Tampermonkey userscript for agnai.chat:
  - Captures LLM responses from DOM
  - Sends to Qwen (LM Studio, 127.0.0.1:1234) for TTS prompt
  - Calls local TTS HTTP server
  - Plays audio with volume slider, play/pause/stop controls
- Fixed: panel position (top not bottom), character card extraction from DOM
- Fixed: only-last-sentence bug → full message sent as single generation
- Fixed: audio looping → trackbar, no loop
- Fixed: voice naming (gongitsune → gogitsune)
- Added: vertical volume slider, generate/play/cancel button separation

### Oct 7–8: TTS Engine Swaps
- Attempted OuteTTS (Llama-OuteTTS-1.0-1B-FP16.gguf) — too slow
- Switched to Kokoro — voices all sounded the same (audiobook-trained)
- User requested: custom voices from LLM description (zero-shot)
- Chose IndexTTS 2.5 for zero-shot voice creation capability

### Oct 9: IndexTTS 2.5 Attempts
- Read `indextts2.5.md` guide
- Attempted to run IndexTTS via dotnet — crashed (0xc0000005)
- Tried via koboldcpp TTS server — 500 errors, timeouts
- Discovered 80-word limit per generation
- Implemented paragraph → sentence segmentation (≤80 words)
- Qwen adds emotion tags per segment
- Result: disjointed output, wrong voice, gaps — abandoned

### Oct 10–13: Pure C Kokoro Server (kokoro-c)
- **Goal:** Headless C TTS server using kokoro.cpp + CUDA on V100
- Read `kokoro-c.md`, `kokoro-solution.md`, `env-info.md`
- Installed: VS 2022, CUDA 12.4.1, Windows SDK
- Built `kokoro-c/` project:
  - `main.c` — server entry, HTTP handling
  - `kokoro.c/h` — model loading, inference
  - `audio.c/h` — WAV I/O, resampling
  - `espeak_phon.c/h` — phonemization
  - `synth.c/h` — synthesis pipeline
  - `chunker.c/h` — text chunking
  - `npz.c/h` — model weight loading
  - `http.c/h` — HTTP server
  - `strings.c/h` — string utilities
  - `common.h` — shared types
- CUDA implementation for V100
- Added: debug tracing (prompt logging before GPU)
- Added: file-input mode (text file → line-by-line TTS)
- Packaged to `C:\bin\kokoro-tts\` with start/stop scripts
- Fixed: WAV fmt/data chunk error
- Fixed: em dash → "dot" pronunciation (text preprocessing)
- Added: Qwen proofreading pass before TTS
- Generated voice samples for all available voices
- Wired to auto-start on user logon
- **Result: Working C TTS server, packaged, integrated with Tampermonkey script**

---

## Phase 2: Spartacvc RVC (Oct 13–17)

### Oct 13: Project Setup + Initial Pipeline
- Created GitHub repo `f-cero/Spartacvc`
- Wrote `ROADMAP.md` (5 phases: assets → core engine → GUI → realtime → virtual mic → Linux)
- Wrote `AGENTS.md` (working protocols: pure C, debug tracing, structure-first, doc everything)
- Downloaded model assets:
  - HuBERT base: `vec-768-layer-12.onnx`, `vec-256-layer-9.onnx`
  - RMVPE: `rmvpe.onnx`
  - Voice models: travisscottv2, beyonce, lana, sabrina (.onnx)
  - Nahida .pth (Genshin JP, 48kHz, 768-dim)
- Built CMake project with ONNX Runtime C API
- Implemented pipeline in C:
  - `src/onnx.c` — ORT session wrapper
  - `src/main.c` — CLI, arg parsing, pipeline orchestration
  - `src/tensor.c/h` — tensor utilities
  - `src/rvc.c/h` — RVC pipeline logic
  - `src/f0.c/h` — RMVPE f0 extraction + cents decode
  - `src/mel.c/h` — mel spectrogram
  - `src/model.c/h` — model config
  - `src/rvc_infer.c/h` — inference orchestration
  - `src/debug.c/h` — SPCV_TRACE() macros
- Fixed ORT C API bugs (4 found via rubber-duck debug):
  1. `Run()` param order wrong
  2. `CastTypeInfoToTensorInfo` ownership (don't double-free)
  3. WAV RIFF chunk header double-read
  4. ONNX elem type 1 = FLOAT(4B) not FLOAT16(2B)
- **E2E working:** af_heart → travisscottv2 (256d, 32kHz) ✓
- **E2E working:** af_heart → nahida (768d, 48kHz) ✓
- Discovered: FPS = sr / prod(upsample_rates) is per-model (not universal)
- Discovered: pitch direction was inverted (low→high should raise f0)
- Set up hf-cli for HuggingFace model downloads
- Wrote `tools/hf-cli.md` reference

### Oct 14: .pth Loading (First Attempt — Lost)
- Read `pickle.md`, `real-pickle.md` (provided by user via Qwen Max)
- Implemented in C:
  - `src/zip_reader.c/h` — ZIP file reader (for .pth container)
  - `src/pickle.c/h` — Python pickle parser
  - `src/pth_loader.c/h` — .pth model loader
  - `src/manifest.c/h` — manifest-based tensor storage
- Created `tools/pth2manifest.py` — one-time .pth → .spacvc converter
- Validated: 457 tensors extracted, 0 mismatches
- **AGENT MISTAKE: Deleted working code during "cleanup" (commit feeee4b)**
- Rebuilt from scratch, lost working state
- Started `pth2onnx` approach: C tool to convert .pth → .onnx
  - `src/onnx_emit.c/h` — ONNX protobuf writer
  - `src/rvc_graph.c/h` — graph emission (arena, constants)
  - `src/rvc_graph_bin.c` — binary graph reader + node emission
  - `src/pth2onnx.c` — CLI: .pth → .onnx
  - `tools/gen_bin.py` — ONNX → binary serializer (dev tool)
  - `src/graph_data.bin` + `src/graph_data.h` — embedded 597KB graph binary
- Built `pth2onnx.exe` (763KB)
- Fixed: heap allocation for pth_model_t (600KB struct, not stack)
- Fixed: fp16→fp32 conversion
- Fixed: opset 16 / IR version 8 (must match reference)
- Fixed: INT32 vs FLOAT constant classification (dtype-aware)
- Fixed: null-terminate all strings from binary reader
- ONNX checker passes for generated model (4938 nodes, 457 initializers)
- Discovered: graph structure is model-specific (kernel sizes vary)
- Created `tools/export_pth.py` — Python export path for different architectures
- Validated: nahida .pth → ONNX → ORT inference → 48kHz output ✓
- Committed and pushed to GitHub

### Oct 15–17: Debugging + Test Failures
- Attempted to run standard test (af_heart → nahida) via `spartacvc.exe`
- Hit: "FATAL: HuBERT load" error (wrong model path: `models/ozada/` vs `models/base/`)
- Looped on `ls` commands instead of debugging
- Used Python unnecessarily to check WAV properties
- Output files from earlier runs were clipping/garbage
- User wiped project: deleted all code, models, binaries
- **Final state: only markdown docs remain**

---

## Key Technical Learnings

1. **ORT C API ownership:** `CastTypeInfoToTensorInfo` returns a VIEW. Free via `ReleaseTypeInfo(ti)` only, NOT `ReleaseTensorTypeAndShapeInfo`.
2. **ORT Run() signature:** `(session, run_opts, in_names, in_values, in_len, out_names, OUT_LEN, out_values)` — output_names before outputs.
3. **ONNX tensor types:** 1=FLOAT(4B), 2=UINT8(1B), 6=INT32(4B), 7=INT64(8B), 10=FLOAT16(2B). NOT 1=FLOAT16.
4. **WAV RIFF:** chunk header = 4B name + 4B size. Don't double-read.
5. **Per-model FPS:** `fps = sr / prod(upsample_rates)`. Nahida: 48000/480=100fps. Ozada 32k: 32000/400=80fps.
6. **Model-specific graphs:** `upsample_kernel_sizes` vary per model. Can't use one graph for all.
7. **MSVC compound literal ICE:** >4000 lines of compound literals crashes the compiler. Use binary data + C runtime instead.
8. **Pickle format:** .pth = ZIP containing `archive/data/*` (tensors) + `archive/data.pkl` (structure). TensorProto protobuf inside.
9. **Protobuf raw_data:** field 9 in TensorProto. Must size correctly by dtype.
10. **Constant classification:** By dtype (data_type field), NOT by byte length. INT32 and FLOAT are both 4 bytes.

---

## Phase 5: GGML/Vulkan + Tray App (Oct 27)

### GGML Pipeline Complete (all verified)
- **Encoder** through ggml ops on CUDA (PG500): maxabs=0.000356 PASS
- **Flow** through ggml conv_1d on CUDA: CPU vs CUDA maxabs=0.0017 PASS
- **Decoder** through ggml ops on CUDA: all 4 stages PASS
- **Full pipeline** integrated into `spartacvc.exe`: 2.4s for 4.1s audio (9x faster than pure-C)
- **Vulkan backend** (AMD Vega FE): STT verified, 13.8s for 4.1s audio
- **`--voice model.gguf`** single-file mode works (no .pth needed at runtime)
- **172s → 2.5s** perf fix (encoder was the bottleneck, now on GPU)

### Tray App (working, Oct 29)
- `src/tray.c` — Win32 tray icon, full menu (backend/model/devices/convert/start-stop/quit)
- `src/tray_main.c` — WinMain entry point
- `src/wasapi.h/.c` — WASAPI capture/render + device enumeration (COM in C, vtable calls)
- `src/pipeline_rt.h/.c` — real-time worker (batch RVC via CLI, ~2-3s latency)
- `src/com_guids.c` — manual COM GUID definitions (SDK lib doesn't export them)
- `src/spartacvc.ico` — placeholder icon (S on dark bg)
- `src/spartacvc.res` — compiled icon resource (276KB)
- `build_tray.bat` — builds `spartacvc_tray.exe` (203MB static)
- **All strings wide/UTF-16** for international release
- **WORKING:** backend selection, model listing, device enumeration, Convert file (via CLI), Start/Stop (batch RVC), config persistence
- **NOT YET:** true real-time RVC (needs pipeline linked directly, not CLI round-trip)

### Tray App Bugs Fixed (Oct 29)
- `eDataFlow` → `EDataFlow` (capital E in SDK)
- `IMMDevice_GetProperty` doesn't exist → use `OpenPropertyStore` + `IPropertyStore::GetValue`
- `WAVEFORMATEXTENSIBLE` cast fails → just report base format, WASAPI auto-converts
- COM GUIDs (`IID_*`, `CLSID_*`) not in mmdevapi.lib → manual definitions in `com_guids.c`
- `ShellBrowseForFolderW` not available → removed browse, rescan instead
- `MB_ICONCHECK` doesn't exist → `MB_ICONINFORMATION`
- `pkt_size` is `UINT64` not `DWORD`
- `GetProperty` takes `const PROPERTYKEY*` not `const DEVPROPKEY*`
- `wasapi_co_init/uninit` were declared but never implemented
- `do_convert_file` needed full path to spartacvc.exe + hubert/rmvpe args
- rt_worker output buffer was `chunk_samples * 480` (way too big) → `chunk_samples`

### What's Next
1. **True real-time RVC** — link pipeline code into tray app (refactor ggml for model reuse)
2. **Model directory browse** — proper folder picker (needs comctl32 v6 manifest)
3. **Virtual mic** — output to VB-Cable or WASAPI virtual device
4. **Icon** — replace placeholder with proper design
5. **i18n** — string table for multiple languages (all strings already wide/UTF-16)
6. **Auto-start** — optionally launch on Windows logon

### Key Files
| File | Purpose |
|------|---------|
| `src/tray.c` | Tray icon, menu, message loop |
| `src/tray_main.c` | WinMain entry |
| `src/wasapi.c` | WASAPI capture/render (COM) |
| `src/pipeline_rt.c` | Real-time worker thread |
| `src/ggml_pipeline.c` | Flow+decoder on GPU (ggml) |
| `src/ggml_rvc_enc_gpu.c` | Encoder on GPU (ggml) |
| `src/gguf_load.c` | GGUF → ggml tensor loading |
| `src/main.c` | CLI entry (file conversion) |
| `build_tray.bat` | Build tray exe |
| `build_spcv.bat` | Build CLI exe |

### Build Commands
```bash
# CLI (full pipeline, file conversion)
cmd //c build_spcv.bat

# Tray app (WASAPI + menu)
cmd //c build_tray.bat

# Run CLI
spartacvc.exe input.wav --voice models/voices/nahida-jp.gguf --backend cuda --out out.wav

# Run tray
spartacvc_tray.exe
```

### GPU Status
| GPU | Backend | Status |
|-----|---------|--------|
| Tesla PG500-216 (V100 lite, sm_70, TCC, 32GB) | CUDA | ✅ 2.4s/4.1s |
| Radeon Vega Frontier Edition | Vulkan | ✅ 13.8s/4.1s |
| CPU (48 cores) | ggml-cpu | ✅ works |

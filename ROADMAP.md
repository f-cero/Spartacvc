# Spartacvς Roadmap

Phases are ordered by dependency. Each phase ends with something runnable and
verifiable — no phase ships "invisible" work.

---

## Phase 0 — Model assets + validated reference (DONE)

- [x] **HuBERT base** (768-dim + 256-dim) → `models/ozada/vec-{768,256}-layer-{12,9}.onnx`
- [x] **RMVPE** f0 extractor → `models/ozada/rmvpe.onnx`
- [x] **RVC v2 voice models** (ONNX) → `models/ozada/{travisscottv2,beyonce,lana,sabrina}.onnx`
- [x] **Nahida .pth** (Genshin JP, 48kHz, 768-dim) → `C:/temp/nahida_extracted/nahida-jp.pth`
- [x] **Working Python reference** → `tools/reference_pipeline.py` (validated E2E)
- [x] **Frame-rate bug found & fixed** — each model has its own native rate:
      `fps = sr / prod(upsample_rates)`. Nahida: 48000/480 = **100fps**.
      Ozada 32k models: 32000/400 = **80fps**. Must read per-model config.
- [x] **Pitch direction confirmed** — with correct frame rate, low→high voice
      conversion raises f0 as expected (af_heart 195Hz → Nahida 202Hz)

**Pipeline (validated):**
```
input WAV → resample 16kHz
  → HuBERT (50fps, [1,T50,768]) → resample to model_fps ([1,T,768])
  → mel(n_fft=1024, hop=160, n_mels=128, fmin=30, fmax=8000, log, pad→mult32)
  → RMVPE → salience [T,360] → cents decode (argmax+9bin avg, thresh 0.05)
  → f0 Hz → resample to model_fps
  → pitch = mel-scale int [1,255], pitchf = raw f0 Hz
  → Voice decoder (phone + pitch + pitchf + ds=0 + rnd=randn(1,192,T)) → audio @sr
```

**Done when:** ✅ Python reference produces correct output for multiple voices.

---

## Phase 1 — Core inference engine (C, no Python at runtime)

The engine: audio in → waveform out. No UI, no audio devices.

### 1A — ONNX path (pre-exported models)

- [x] CMake project, fetch ONNX Runtime C API via `FetchContent`
- [x] Execution-provider selection:
      - **Windows: DirectML → CPU** (code in place; needs DML-enabled ORT build)
      - **Linux: Vulkan Compute → CPU** (TODO — Phase 5)
      - **No CUDA, no ROCm** — keeps binary lean, no 100MB+ DLL bloat
- [x] Per-model frame rate: derived from voice model (256d→32k/80fps, 768d→48k/100fps)
- [x] Pipeline: resample→16k → HuBERT → resample→model_fps →
      RMVPE → f0 → resample→model_fps → pitch shift → voice decoder → output
- [x] Pitch shift: `f0 * 2^(semitones/12)` — simple, effective, user-controlled
- [x] Config file (JSON): `voice_model`, `hubert_model`, `rmvpe_model`,
      `pitch_shift_semitones`, `backend`, `output_dir`
- [x] Offline CLI: `spartacvc in.wav --voice <v> --hubert <h> [--config <c>]`
- [x] Nahida (768d) E2E: af_heart → Nahida @ 48kHz/100fps ✓
- [x] Beyoncé (256d) E2E: af_heart → Beyoncé @ 32kHz/80fps ✓

### 1B — .pth model loading (manifest-based)

The user drops a `.pth` file. The app converts it to a `.spacvc` manifest
directory on first load, then reads the manifest directly.

- [x] **`pth2manifest.py`** — one-time converter: `.pth` → `.spacvc/`
      (manifest.bin + data/0..N). Extracts tensor names, shapes, dtypes,
      config. Validated: 457 tensors, 0 mismatches, values match Python.
- [x] **`manifest.h/c`** — C manifest reader. Loads manifest.bin, reads
      tensor data from data/ files. Heap-allocated (1.4MB struct).
- [ ] **`onnx_builder.c`** — instantiate the known RVC v2 architecture
      (`SynthesizerTrnMs{256,768}NSFsid`) as an ONNX protobuf graph from
      manifest tensor data. Mechanical: walk the layer list, emit
      Conv/MatMul/Add/LayerNorm/Attention nodes, set initializers.
- [ ] **`model_loader.c`** — dispatch by extension:
      - `.onnx` → load directly into ONNX Runtime
      - `.pth` → pth2manifest → manifest reader → onnx_builder → cache as `.spacvc`
- [ ] First-load converts + caches; subsequent loads read manifest directly

**Done when:** `spartacvc in.wav -o out.wav --model nahida-jp.pth` works with
zero pre-processing, on DirectML (Windows) and CPU. Second run loads from cache.

---

## Phase 2 — Config file + Win32 config window + tray

The "double-click the exe" experience.

- [ ] Read/write `config.json` next to the exe (plain JSON, hand-editable)
- [ ] Win32 config window: model picker (scans models/ for .onnx + .pth),
      input device, output device, pitch shift slider (-24 to +24 semitones),
      index/filter/rms/protect sliders, backend selector, Start/Stop
- [ ] System tray icon (`Shell_NotifyIcon`): start/stop, open config, quit
- [ ] No console window in GUI mode (Windows subsystem)

**Done when:** launch exe → window → pick a model + devices → start → tray.

---

## Phase 3 — Real-time voice-changer mode

- [ ] WASAPI capture from the selected input device (shared mode, 48 kHz)
- [ ] Chunked pipeline (≈ 120 ms frames) with a small ring buffer
- [ ] WASAPI render to the selected output device
- [ ] Latency budget: **< 100 ms** end-to-end on GPU; report measured latency
- [ ] Graceful device hot-unplug handling

**Done when:** speak into the mic, hear the converted voice out the output
device in real time, stable for 10+ minutes.

---

## Phase 4 — "RVC-output" virtual microphone

The converted voice must appear as an *input* device to other apps.

- [ ] **Option A (MVP):** document + auto-detect a virtual cable
      (VB-Cable / VoiceMeeter / Steam Audio) as the output device; the cable's
      "Input" side is the mic other apps see. Zero driver code.
- [ ] **Option B (stretch):** bundle a small open-source WASAPI virtual audio
      driver so Spartacvς *is* the "RVC-output" device, no third-party cable.

**Done when:** set Spartacvς output to the virtual cable → select the cable input
in Discord/a game → the *converted* voice is what they hear.

---

## Phase 5 — Linux port + packaging

- [ ] **Vulkan Compute EP** for GPU inference (universal: AMD, NVIDIA, Intel,
      old cards, new cards — anything with Vulkan 1.2+ compute shaders)
      - If ONNX Runtime's Vulkan EP is immature: write custom Vulkan kernels
        for the hot ops (Conv1d, MatMul, Attention, LayerNorm)
      - Fallback: CPU EP (valid for server / low-power use cases)
- [ ] Linux audio backend (PulseAudio / JACK / ALSA)
- [ ] Single-folder distribution: `spartacvc` + `libonnxruntime.so` + `models/`
- [ ] CI: build matrix (Windows x64, Linux x64), smoke test
- [ ] Tag `v0.1.0`

**Done when:** a stranger can clone, build, drop in a .pth model, and convert
audio on Windows or Linux following only the README.

---

## Execution provider strategy

| Platform | Primary | Fallback | Notes |
|----------|---------|----------|-------|
| Windows | **DirectML** | CPU | Any D3D12 GPU (AMD, NVIDIA, Intel). No CUDA. |
| Linux | **Vulkan Compute** | CPU | Any Vulkan 1.2+ GPU. No CUDA, no ROCm. Old cards welcome. |
| Anywhere | CPU | — | Valid for server / offline / low-power. |

**Why no CUDA:**
- Inflates binary 100MB+ (cudart, cublas, cudnn DLLs)
- DirectML already covers NVIDIA on Windows at near-CUDA speed for inference
- Vulkan covers NVIDIA on Linux without vendor SDK
- Keeps the project vendor-agnostic and the binary lean

**Why no ROCm:**
- Whitelist of supported GPUs excludes older cards (Vega FE, etc.)
- Vulkan is the universal floor — if it has compute shaders, it runs

---

## Non-goals (for now)

- Training / fine-tuning models (use existing RVC tools for that)
- GPT-SoVITS / other TTS families (different architecture)
- macOS (Metal / CoreAudio backend is a natural later addition)
- General-purpose PyTorch→ONNX converter (we instantiate known architectures only)

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| ONNX Runtime Vulkan EP immature on Linux | Custom Vulkan kernels for hot ops; CPU fallback always works |
| DirectML op coverage gap on some RVC op | CPU fallback for that op; file upstream; avoid exotic ops |
| .pth format edge cases | Manifest converter handles all PyTorch ZIP+pickle; C reads simple binary manifest |
| Real-time latency on CPU | Label CPU as "offline-first"; GPU recommended for real-time |
| Model architecture variants (v1 vs v2, 256 vs 768 dim) | Config-driven: read dim from .pth, instantiate correct class |
| Virtual-driver (Phase 4B) complexity | Ship Option A first; 4B is optional stretch |

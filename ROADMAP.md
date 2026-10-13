# Spartacvς Roadmap

Phases are ordered by dependency. Each phase ends with something runnable and
verifiable — no phase ships "invisible" work.

---

## Phase 0 — Model assets (unblock everything)

RVC inference needs three ONNX graphs. Torch `.pth` is the community format, so
we either find pre-exported ONNX or do a **one-off** `.pth → ONNX` export
(throwaway Python env — never a runtime dependency).

- [x] **HuBERT base** (768-dim features) → `models/ozada/vec-768-layer-12.onnx`
- [x] **RMVPE** f0 extractor → `models/ozada/rmvpe.onnx`
- [x] **One RVC v2 voice model** → `models/ozada/travisscottv2.onnx`
- [x] **Working Python reference** → `tools/reference_pipeline.py` (validated E2E)
- [ ] Port reference to C (ONNX Runtime C API) + confirm DirectML session runs

**Pipeline (validated):** resample→16k → HuBERT (50fps) → **resample 80fps** →
RMVPE mel(128,n_fft1024,hop160,fmin30,fmax8k)→cents-decode→f0 → resample 80fps →
World decoder `phone[1,T,768]`+`pitch`+`pitchf`+`ds`+`rnd[1,192,T]` → audio @32k.
Voice model native rate = **80fps** (400 samples/frame @32k); output length = input.

**Done when:** a reference C program loads all three ONNX files and converts a
test WAV with audible, correct output.

---

## Phase 1 — Core inference library (portable C)

The engine, no UI, no audio devices. Pure pipeline in + waveform out.

- [ ] CMake project, fetch ONNX Runtime (C API) via `FetchContent`
- [ ] Execution-provider selection: **DirectML → CUDA → CPU**, auto or forced
- [ ] Pipeline: resample → f0 (RMVPE) → features (HuBERT) → convert (voice) →
      pitch shift → RMS mix → protect → resample out
- [ ] Config struct (JSON) with all RVC knobs:
      `f0_up_key`, `index_rate`, `filter_radius`, `rms_mix_rate`, `protect`,
      `f0_method`, `backend`, `models_dir`
- [ ] Offline CLI: `spartacvc in.wav -o out.wav --config config.json`

**Done when:** `spartacvc in.wav -o out.wav` produces a correct conversion on the
V100 (DirectML) and on CPU fallback, from a clean build.

---

## Phase 2 — Config file + Win32 config window + tray

The "double-click the exe" experience.

- [ ] Read/write `config.json` next to the exe (plain JSON, hand-editable)
- [ ] Win32 config window: model picker, input device, output device,
      pitch/index/filter/rms/protect sliders, backend selector, Start/Stop
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
      driver so `Spartacvς` *is* the "RVC-output" device, no third-party cable.
      (Larger effort — kernel driver + installer; only after A works.)

**Done when:** set Spartacvς output to the virtual cable → select the cable input
in Discord/a game → the *converted* voice is what they hear.

---

## Phase 5 — Portability, packaging, docs

- [ ] Linux audio backend (ALSA/Pulse) + CUDA EP path
- [ ] Single-folder distribution: `spartacvc.exe` + `onnxruntime.dll` + `models/`
- [ ] CI: build matrix (Windows x64, Linux x64), smoke test
- [ ] `tools/download_models.md` finalized; first-contribution guide
- [ ] Tag `v0.1.0`

**Done when:** a stranger can clone, build, drop in a model, and convert audio
on Windows or Linux following only the README.

---

## Non-goals (for now)

- Training / fine-tuning models (use existing RVC tools for that)
- GPT-SoVITS / other TTS families (different architecture)
- macOS (CoreAudio backend is a natural later addition, not a launch blocker)

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| ONNX export of a `.pth` model differs from torch output | Phase 0 reference-match gate before any C work |
| DirectML op coverage gap on some RVC op | Fall back to CUDA/CPU EP; file upstream; avoid exotic ops in export |
| Real-time latency on CPU | Clearly label CPU as "offline-first"; GPU recommended for real-time |
| Virtual-driver (Phase 4B) complexity | Ship Option A first; 4B is optional stretch |

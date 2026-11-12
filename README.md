# Spartacvς — voice conversion in pure C

**RVC (Retrieval-based Voice Conversion)** for Windows with no Python and no
huge runtime stack: a native C engine, a Win32 tray app, and its own virtual
microphone driver. Voice models are single GGUF files.

RVC swaps the *timbre* of speech (who is speaking) while preserving *prosody*
(how it is spoken) — which is why it pairs perfectly with a TTS engine like
Kokoro: the TTS brings the emotion and pacing, Spartacvς brings the voice.

## Status

**Working now:**

- **Offline conversion (CLI)** — WAV in, converted WAV out. Measured on a
  Tesla V100 (CUDA): **2.4 s for 4.1 s of audio**. AMD Radeon Vega FE
  (Vulkan): 13.8 s. CPU: works everywhere.
- **Tray app** (`spartacvc_tray.exe`) — model picker, input/output device
  picker, runtime (backend) selector, pitch submenu, batch file conversion,
  load-state tray icons, config persistence.
- **Virtual microphone driver** (`Spartacvc Virtual Audio`) — a bundled KMDF
  driver that creates two audio endpoints:
  - **Spartacvc Mic** — a capture device other apps (Discord, games) can select
  - **Spartacvc (RVC output)** — a render device Spartacvc plays the converted
    voice into
- **Single-file voice models** — drop in a `.pth` and Spartacvc converts it to
  GGUF in-app (in memory, parallelized); or use a pre-converted `.gguf`.

**In progress:**

- **Real-time conversion** — currently the tray drives the CLI per chunk
  (~2–3 s latency). Work to link the pipeline directly into the tray app
  (<100 ms budget on GPU) is the current frontier; see
  [current-state.md](current-state.md).

## How it works

```
input audio (WAV or capture device)
   HuBERT (speaker embedding, ONNX Runtime)
   RMVPE  (pitch/f0 extraction, ONNX Runtime)
   pitch shift (−12…+12 semitones)
   RVC voice decoder — encoder, flow, decoder on ggml
output audio (WAV or render device)
```

Backends (auto-detected, selectable in the tray or via CLI):

| Concern        | Answer                                                 |
| -------------- | ------------------------------------------------------ |
| GPU (NVIDIA)   | **CUDA** backend (ggml-cuda)                           |
| GPU (AMD/any)  | **Vulkan** backend (ggml-vulkan) — any Vulkan 1.2+ GPU |
| No GPU         | **CPU** fallback (ggml-cpu) — always works             |
| ONNX models    | ONNX Runtime (DirectML/CPU) for HuBERT + RMVPE         |
| Language       | Pure C runtime + WASAPI audio + Win32 tray             |

## Building

```bat
cmd /c build_spcv.bat    :: CLI        -> spartacvc.exe
cmd /c build_tray.bat    :: Tray app   -> spartacvc_tray.exe
```

Requirements: Visual Studio 2022 (C++ tools), CMake, CUDA toolkit (optional,
for the CUDA backend), Vulkan SDK (optional, for the Vulkan backend).

### Virtual audio driver

The driver lives in `third_party/virtual-audio-driver` (KMDF PortCls sample
derivative). Building it needs the WDK; installing it needs test signing
enabled (`bcdedit /set testsigning on`) until it ships with attestation
signing. KMDF note: build against **KMDF 1.15** headers/stubs for Windows 10
— later KMDF stub versions fail the loader bind on 19041 (see
[current-state.md](current-state.md) §4 for the full story).

## Repository layout

```
Spartacvc/
├── README.md / ROADMAP.md / PROMPT.md / AGENTS.md
├── current-state.md         # detailed handoff/status doc
├── build_spcv.bat           # build CLI (spartacvc.exe)
├── build_tray.bat           # build tray app (spartacvc_tray.exe)
├── models/
│   ├── base/                # HuBERT + RMVPE (.onnx)
│   └── voices/              # voice models (.gguf / .pth)
├── src/                     # everything is pure C
│   ├── main.c               # CLI entry point
│   ├── tray.c / tray_main.c # Win32 tray app
│   ├── wasapi.c             # WASAPI capture/render + device enum
│   ├── pipeline_rt.c        # real-time worker thread
│   ├── ggml_pipeline.c      # RVC flow+decoder on GPU (ggml)
│   ├── ggml_rvc_enc*.c      # RVC encoder on GPU (ggml)
│   ├── gguf_load.c          # GGUF → ggml tensor loading
│   ├── pth_loader.c         # .pth (ZIP+pickle) reading
│   ├── pth2gguf.c           # .pth → GGUF converter
│   ├── onnx*.c              # ONNX Runtime wrapper (HuBERT, RMVPE)
│   ├── mel.c / f0.c         # mel spectrogram / f0 extraction
│   ├── faiss_index.c        # FAISS retrieval index
│   └── ...
├── tools/                   # dev-time scripts (reference, spectrograms, STT)
└── third_party/
    ├── onnxruntime/         # ONNX Runtime C API (HuBERT/RMVPE)
    └── virtual-audio-driver/ # KMDF virtual mic/speaker driver
```

## Getting models

Spartacvς does **not** ship voice models (they are third-party works with
their own licenses). Point the app at a models folder; see
[`tools/download_models.md`](tools/download_models.md) for sources. `.pth`
models are converted to GGUF automatically in-app; `.gguf` models load
directly.

## License

[Apache License 2.0](LICENSE) — use it, modify it, ship it, sell it. Keep the
attribution, don't hold the authors liable. That's it.

---

*vibe-coded by fluffy, using Qwen 3.8 27B (Q4_k_m) on a Tesla V100. The GPU did
the math; the vibes did the rest.* 🤖

# Spartacvς — voice conversion in pure C

Real-time and offline **RVC (Retrieval-based Voice Conversion)** in a single native
executable. No Python. No runtime dependencies. One exe, one models folder.

RVC swaps the *timbre* of speech (who is speaking) while preserving *prosody*
(how it is spoken) — which is why it pairs perfectly with a TTS engine like
Kokoro: the TTS brings the emotion and pacing, Spartacvς brings the voice.

## Why C?

Every existing RVC runtime is a Python stack (torch + fairseq + faiss), which on
Windows means source-building abandoned libraries or running Docker. The models
themselves are just neural nets — they run fine as ONNX graphs. Spartacvς executes
them with **ggml** (the tensor library from llama.cpp) — a single C library
with pluggable GPU backends:

| Concern        | Answer                                                        |
| -------------- | ------------------------------------------------------------- |
| GPU (NVIDIA)   | **CUDA** backend (ggml-cuda)                                  |
| GPU (AMD/any)  | **Vulkan** backend (ggml-vulkan) — any Vulkan 1.2+ GPU        |
| No GPU         | **CPU** fallback (ggml-cpu) — always works                    |
| Backend choice | Auto-detected at startup, override in tray menu or CLI        |
| Language       | Pure C runtime + WASAPI audio + Win32 tray                    |

## Modes

**Real-time (voice changer).** Captures from any input device you pick, converts
every chunk, and renders to any output device. Point the output at a virtual
cable (VB-Cable, VoiceMeeter, Steam Audio…) and the converted voice shows up in
every other app as a microphone — games, Discord, RP, streaming. Target latency:
under 100 ms end-to-end on a modern GPU.

**Offline (batch).** Convert WAV files from the command line or the UI.

**Tray app.** Double-click the exe → config window → start → minimize to the
system tray. Config lives in a plain-text file next to the exe; edit it by hand
or through the window.

## Features (target)

- RVC v1 + v2 models (`.pth`-derived ONNX graphs + optional `.index` retrieval)
- Pitch control (`f0_up_key`), index rate, filter radius, RMS mix, noise protect
- f0 extraction: RMVPE (ONNX) — the modern RVC default
- Execution-provider priority: DirectML → CUDA → CPU (auto or forced)
- Per-model profiles (pitch, index rate, etc.) saved in config
- Device picker for input and output (Windows: WASAPI enumeration)
- System tray with start/stop, no console window in normal use

## Repository layout

```
$spartacvc/
├── README.md
├── ROADMAP.md
├── PROMPT.md                # master project brief
├── AGENTS.md                # agent rules
├── build_spcv.bat           # build CLI (spartacvc.exe)
├── build_tray.bat           # build tray app (spartacvc_tray.exe)
├── models/
│   ├── base/                # HuBERT + RMVPE (.onnx)
│   └── voices/              # voice models (.gguf / .pth)
├── src/
│   ├── main.c               # CLI entry point
│   ├── tray.c               # Win32 system tray app
│   ├── tray_main.c          # WinMain entry
│   ├── wasapi.c             # WASAPI capture/render + device enum
│   ├── pipeline_rt.c        # real-time worker thread
│   ├── ggml_pipeline.c      # RVC flow+decoder on GPU (ggml)
│   ├── ggml_rvc_enc_gpu.c   # RVC encoder on GPU (ggml)
│   ├── gguf_load.c          # GGUF → ggml tensor loading
│   ├── rvc.c                # RVC model config
│   ├── onnx.c               # ONNX Runtime wrapper (HuBERT, RMVPE)
│   ├── mel.c                # mel spectrogram
│   ├── f0.c                 # f0 extraction
│   ├── faiss_index.c        # FAISS retrieval index
│   ├── pth_loader.c         # .pth model loading
│   ├── pth2gguf.c           # .pth → GGUF converter
│   ├── com_guids.c          # manual COM GUID definitions
│   └── ...
├── tools/
│   ├── stt.py               # STT for "listening" to output
│   ├── spectrogram.py       # spectrogram generation
│   └── download_models.md   # where to get models
└── third_party/
    └── onnxruntime/         # ONNX Runtime C API (for HuBERT/RMVPE)
```

## Getting models

Spartacvς does **not** ship voice models (they are third-party works with their own
licenses). Point the config at a models folder; see
[`tools/download_models.md`](tools/download_models.md) for verified sources and
how to convert a `.pth` RVC model to the ONNX layout Spartacvς expects.

## Status

🚧 **In development** — see [ROADMAP.md](ROADMAP.md) for the phase plan.

## License

[Apache License 2.0](LICENSE) — use it, modify it, ship it, sell it. Keep the
attribution, don't hold the authors liable. That's it.

---

*vibe-coded by fluffy, using Qwen 3.8 27B (Q4_k_m) on a Tesla V100. The GPU did
the math; the vibes did the rest.* 🤖

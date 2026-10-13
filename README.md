# Spartacvς — voice conversion in pure C

Real-time and offline **RVC (Retrieval-based Voice Conversion)** in a single native
executable. No Python. No runtime dependencies. One exe, one models folder.

RVC swaps the *timbre* of speech (who is speaking) while preserving *prosody*
(how it is spoken) — which is why it pairs perfectly with a TTS engine like
Kokoro: the TTS brings the emotion and pacing, Spartacvc brings the voice.

## Why C?

Every existing RVC runtime is a Python stack (torch + fairseq + faiss), which on
Windows means source-building abandoned libraries or running Docker. The models
themselves are just neural nets — they run fine as ONNX graphs. `Spartacvc` executes
them with the ONNX Runtime C API and nothing else:

| Concern        | Answer                                                        |
| -------------- | ------------------------------------------------------------- |
| GPU (Windows)  | **DirectML** — any DirectX 12 GPU, old or new                 |
| GPU (NVIDIA)   | CUDA execution provider (when available)                      |
| No GPU         | CPU fallback — always works, slower                           |
| Backend choice | Auto-detected at startup, override in config                  |
| Language       | C (portable core) + platform audio backends (WASAPI first)    |

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

## Repository layout (planned)

```
$spartacvc/
├── CMakeLists.txt
├── LICENSE                  # Apache-2.0
├── README.md
├── ROADMAP.md
├── config.example.json
├── src/
│   ├── main.c               # entry: CLI or GUI
│   ├── core/                # portable C: pipeline, config, model loading
│   ├── audio/               # resample, rms, mix, wav i/o
│   ├── backends/
│   │   ├── ort.c            # ONNX Runtime C API wrapper
│   │   └── ep.c             # execution-provider selection (DirectML/CUDA/CPU)
│   ├── platform/
│   │   ├── win/             # WASAPI capture/render, Win32 UI, tray
│   │   └── unix/            # ALSA/Pulse (later)
│   └── ui/                  # config window (Win32)
├── tools/
│   └── download_models.md   # where to get models + how to add your own
└── third_party/             # onnxruntime (fetched via CMake, not vendored)
```

## Getting models

Spartacvc does **not** ship voice models (they are third-party works with their own
licenses). Point the config at a models folder; see
[`tools/download_models.md`](tools/download_models.md) for verified sources and
how to convert a `.pth` RVC model to the ONNX layout Spartacvc expects.

## Status

🚧 **In development** — see [ROADMAP.md](ROADMAP.md) for the phase plan.

## License

[Apache License 2.0](LICENSE) — use it, modify it, ship it, sell it. Keep the
attribution, don't hold the authors liable. That's it.

---

*vibe-coded by fluffy, using Qwen 3.8 27B (Q4_k_m) on a Tesla V100. The GPU did
the math; the vibes did the rest.* 🤖

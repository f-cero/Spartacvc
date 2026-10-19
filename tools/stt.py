#!/usr/bin/env python3
"""STT tool: transcribe a WAV file to text (objective "listen").

Usage:
    python tools/stt.py <wav> [model_size]

model_size: tiny|base|small|medium (default: base)
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    wav = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "base"
    if not os.path.exists(wav):
        print(f"ERROR: {wav} not found")
        sys.exit(1)
    print(f"Loading faster-whisper model '{size}' ...")
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device="cpu", compute_type="int8")
    # Load audio ourselves (av has API mismatch); resample to 16k mono
    import numpy as np
    from scipy.io import wavfile
    sr, data = wavfile.read(wav)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    if sr != 16000:
        from scipy.signal import resample_poly
        g = np.gcd(sr, 16000)
        data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
    print(f"Transcribing {wav} ({len(data)/16000:.2f}s) ...")
    segments, info = model.transcribe(data, beam_size=5, vad_filter=True)
    print(f"  language={info.language} (p={info.language_probability:.2f}) "
          f"duration={info.duration:.2f}s")
    text_parts = []
    for seg in segments:
        print(f"  [{seg.start:6.2f} - {seg.end:6.2f}] {seg.text.strip()}")
        text_parts.append(seg.text.strip())
    print("=== TRANSCRIPT ===")
    print(" ".join(text_parts).strip() if text_parts else "(no speech detected)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
spartacvc — RVC voice converter CLI
vibe-coded by fluffy using Qwen 3.8 27b Q4_k_m on Tesla V100

Usage:
    python spartacvc.py input.wav -o output.wav --voice model.onnx [--hubert hubert.onnx] [--pitch +12]

Models (auto-detected from ONNX input names):
    256-dim: beyonce.onnx, sabrina.onnx  (HuBERT vec-256-layer-9, 80fps, 32kHz)
    768-dim: lana.onnx, travisscottv2.onnx (HuBERT vec-768-layer-12, 80fps, 32kHz)
"""
import argparse, os, sys, time
import numpy as np
import soundfile as sf
import onnxruntime as ort
from scipy.signal import resample

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ozada")
CPU = ["CPUExecutionProvider"]

def load_sessions(voice_path, hubert_path, rmvpe_path):
    hub = ort.InferenceSession(hubert_path, providers=CPU)
    rmvpe = ort.InferenceSession(rmvpe_path, providers=CPU)
    voice = ort.InferenceSession(voice_path, providers=CPU)
    return hub, rmvpe, voice

def detect_fps(voice_sess):
    """Detect output sample rate from voice model."""
    # RVC v2 models output at 32kHz or 48kHz
    # We'll detect from the model metadata or default to 32kHz
    # For now, check if the model was trained at 48kHz (Nahida) or 32kHz (ozada)
    # The output rate = sr from config, which we can't easily read from ONNX
    # Default to 32kHz for ozada models
    return 32000

def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel2hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def compute_mel(wav16, n_fft=1024, hop=160, n_mels=128, fmin=30, fmax=8000, sr=16000):
    """Compute mel spectrogram."""
    pad = n_fft // 2
    xp = np.pad(wav16, (pad, pad), mode='reflect')
    nf = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(xp[idx] * np.hanning(n_fft), axis=1))
    
    nfbin = n_fft // 2 + 1
    low, high = hz2mel(fmin), hz2mel(fmax)
    mp = np.linspace(low, high, n_mels + 2)
    hp = mel2hz(mp)
    bp = np.floor((n_fft / 2 + 1) * hp / (sr / 2)).astype(int)
    
    fb = np.zeros((n_mels, nfbin), np.float32)
    for i in range(n_mels):
        l, c, r = bp[i], bp[i+1], bp[i+2]
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    
    ms = np.log(np.clip(S @ fb.T, 1e-5, None)).astype(np.float32)
    return ms

def extract_f0(ms, rmvpe_sess):
    """Extract f0 from mel spectrogram using RMVPE."""
    n_mels = ms.shape[1]
    # Pad to multiple of 32
    nff = ms.shape[0]
    npad = 32 * ((nff - 1) // 32 + 1) - nff
    if npad > 0:
        ms = np.pad(ms, ((0, npad), (0, 0)))
    
    sal = rmvpe_sess.run(None, {rmvpe_sess.get_inputs()[0].name: ms.T.reshape(1, n_mels, -1)})[0][0]
    
    # Cents decode with 9-bin averaging
    cm = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
    center = np.argmax(sal, axis=1)
    sp = np.pad(sal, ((0, 0), (4, 4)))
    center += 4
    st = center - 4
    en = center + 5
    ts = np.stack([sp[i, st[i]:en[i]] for i in range(len(center))], 0)
    tc = np.stack([cm[st[i]:en[i]] for i in range(len(center))], 0)
    cents = (ts * tc).sum(1) / ts.sum(1)
    cents[sal.max(1) <= 0.05] = 0
    f0 = 10 * (2 ** (cents / 1200))
    f0[f0 == 10] = 0
    return f0

def resample_to_target(signal, target_len):
    """Linear interpolation resample."""
    if len(signal) == target_len:
        return signal
    idx = np.linspace(0, len(signal) - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(signal) - 1)
    fr = (idx - lo)[:, None] if signal.ndim > 1 else (idx - lo)
    if signal.ndim > 1:
        return signal[lo] * (1 - fr) + signal[hi] * fr
    return signal[lo] * (1 - fr) + signal[hi] * fr

def convert(input_path, output_path, voice_path, hubert_path, rmvpe_path, pitch_shift=0):
    """Full RVC conversion pipeline."""
    t_start = time.time()
    
    # Load audio
    wav, sr = sf.read(input_path)
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    
    # Resample to 16kHz
    wav16 = resample(wav, int(len(wav) * 16000 / sr)).astype(np.float32)
    dur = len(wav16) / 16000
    print(f"Input: {dur:.2f}s @ {sr}Hz -> {len(wav16)} samples @ 16kHz")
    
    # Load models
    print(f"Loading models...")
    hub, rmvpe, voice = load_sessions(voice_path, hubert_path, rmvpe_path)
    
    # Detect HuBERT output dim
    hub_out_dim = 256  # default
    hub_inputs = hub.get_inputs()
    hub_outputs = hub.get_outputs()
    # The output dim is in the shape (last element)
    out_shape = hub_outputs[0].shape
    if len(out_shape) >= 3 and isinstance(out_shape[-1], int):
        hub_out_dim = out_shape[-1]
    print(f"HuBERT dim: {hub_out_dim}")
    
    # Target frame rate: 50fps (HuBERT native) -> resample to model fps
    # For 256-dim models: 80fps (32000/400)
    # For 768-dim models: 80fps (32000/400) or 100fps (48000/480)
    # We'll use 80fps as default (works for ozada models)
    TARGET_FPS = 80.0
    T = int(round(dur * TARGET_FPS))
    print(f"Target: {T} frames @ {TARGET_FPS}fps")
    
    # 1. HuBERT -> phone embeddings
    t0 = time.time()
    embed = hub.run(None, {hub_inputs[0].name: wav16.reshape(1, 1, -1)})[0]  # [1, T50, FEDIM]
    T50 = embed.shape[1]
    e = embed[0]
    if T != T50:
        e = resample_to_target(e, T)
    phone = e.astype(np.float32)[None]
    print(f"HuBERT: {time.time()-t0:.3f}s ({T50}->{T} frames)")
    
    # 2. Mel spectrogram
    t0 = time.time()
    ms = compute_mel(wav16)
    print(f"Mel: {time.time()-t0:.3f}s ({ms.shape[0]} frames)")
    
    # 3. RMVPE -> f0
    t0 = time.time()
    f0 = extract_f0(ms, rmvpe)
    # Resample f0 to target frame rate
    if len(f0) != T:
        f0 = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(f0)), f0)
    print(f"RMVPE: {time.time()-t0:.3f}s (f0 range: {f0[f0>0].min():.1f}-{f0[f0>0].max():.1f} Hz)" if (f0>0).any() else f"RMVPE: {time.time()-t0:.3f}s (no voiced frames)")
    
    # 4. Pitch shift
    if pitch_shift != 0:
        shift = 2 ** (pitch_shift / 12)
        f0 = f0 * shift
        print(f"Pitch shift: {pitch_shift:+d} semitones (×{shift:.3f})")
    
    # 5. Prepare pitch inputs
    pf = f0.astype(np.float32).reshape(1, -1)
    f0min, f0max = 50, 1100
    fmm = 1127 * np.log(1 + f0min / 700)
    fmx = 1127 * np.log(1 + f0max / 700)
    fm = 1127 * np.log(1 + pf / 700)
    fm[fm > 0] = (fm[fm > 0] - fmm) * 254 / (fmx - fmm) + 1
    fm[fm <= 1] = 1
    fm[fm > 255] = 255
    pitch = np.rint(fm).astype(np.int64)
    
    # 6. Voice decoder
    t0 = time.time()
    vi = voice.get_inputs()
    # Input order: phone, length, pitch, pitchf, ds, rnd
    rnd = np.random.randn(1, 192, T).astype(np.float32)
    out = voice.run(None, {
        vi[0].name: phone,
        vi[1].name: np.array([T], np.int64),
        vi[2].name: pitch,
        vi[3].name: pf,
        vi[4].name: np.array([0], np.int64),
        vi[5].name: rnd,
    })[0]
    print(f"Decoder: {time.time()-t0:.3f}s")
    
    # 7. Output
    a = out.reshape(-1)
    out_sr = 32000  # default for ozada models
    out_dur = len(a) / out_sr
    peak = float(np.abs(a).max())
    rms = float(np.sqrt(np.mean(a**2)))
    print(f"Output: {out_dur:.2f}s @ {out_sr}Hz peak={peak:.3f} rms={rms:.4f}")
    
    # Normalize to avoid clipping
    if peak > 0.95:
        a = a * (0.95 / peak)
    
    sf.write(output_path, (a * 32767).astype(np.int16), out_sr)
    print(f"WROTE {output_path} ({os.path.getsize(output_path):,} bytes)")
    print(f"Total: {time.time()-t_start:.2f}s")
    
    # Flatness check
    s = np.abs(np.fft.rfft(a * np.hanning(len(a)))) + 1e-9
    flatness = np.exp(np.mean(np.log(s))) / np.mean(s)
    print(f"Flatness: {flatness:.4f} (speech~0.02-0.1)")
    
    return output_path

def main():
    p = argparse.ArgumentParser(description="spartacvc — RVC voice converter")
    p.add_argument("input", help="Input WAV file")
    p.add_argument("-o", "--output", default=None, help="Output WAV file")
    p.add_argument("--voice", default=None, help="Voice model .onnx")
    p.add_argument("--hubert", default=None, help="HuBERT model .onnx")
    p.add_argument("--rmvpe", default=None, help="RMVPE model .onnx")
    p.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones (-24 to +24)")
    args = p.parse_args()
    
    voice = args.voice or os.path.join(MODEL_DIR, "beyonce.onnx")
    # Auto-select HuBERT based on voice model dim
    if "768" in voice or "lana" in voice or "travis" in voice:
        hubert = args.hubert or os.path.join(MODEL_DIR, "vec-768-layer-12.onnx")
    else:
        hubert = args.hubert or os.path.join(MODEL_DIR, "vec-256-layer-9.onnx")
    rmvpe = args.rmvpe or os.path.join(MODEL_DIR, "rmvpe.onnx")
    
    if not args.output:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"spartacvc_{base}.wav"
    
    print(f"spartacvc v0.1")
    print(f"  voice:  {os.path.basename(voice)}")
    print(f"  hubert: {os.path.basename(hubert)}")
    print(f"  rmvpe:  {os.path.basename(rmvpe)}")
    if args.pitch != 0:
        print(f"  pitch:  {args.pitch:+d} semitones")
    print()
    
    convert(args.input, args.output, voice, hubert, rmvpe, args.pitch)

if __name__ == "__main__":
    main()

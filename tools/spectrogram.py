#!/usr/bin/env python3
"""
spectrogram.py — dev-time diagnostic: render a WAV as a spectrogram PNG.

Usage:
    python tools/spectrogram.py input.wav [out.png] [--fmax 8000] [--nfft 2048]

Writes a labeled spectrogram (time on x, frequency on y, high freq at top).
Horizontal lines = tonal/hum artifacts; smeared vertical energy = consonants;
a clean formant track that changes over time = real, distinct phonemes.
No matplotlib needed — numpy + scipy + PIL only.
"""
import sys
import wave
import numpy as np
from scipy import signal
from PIL import Image, ImageDraw


def load_wav(path):
    w = wave.open(path, "rb")
    sr = w.getframerate()
    ch = w.getnchannels()
    sw = w.getsampwidth()
    n = w.getnframes()
    data = w.readframes(n)
    if sw == 1:
        x = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sw == 2:
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported sample width %d" % sw)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def make_spectrogram(x, sr, nfft=2048, fmax=8000):
    noverlap = nfft - nfft // 4          # 75% overlap
    f, t, Z = signal.stft(x, fs=sr, nperseg=nfft, nfft=nfft,
                          noverlap=noverlap, window="hann")
    mag = np.abs(Z)
    keep = f <= fmax
    f = f[keep]
    mag = mag[keep]
    return f, t, mag


def render(f, t, mag, path, dbmin=-85.0):
    db = 20.0 * np.log10(mag + 1e-9)
    db = np.clip(db, dbmin, 0.0)
    norm = (db - dbmin) / (-dbmin)          # 0..1, 1 = loudest
    # rows = time (top->bottom), cols = freq (low->high)
    spec = norm.T                            # [ntime, nfreq]
    ntime, nfreq = spec.shape

    # Upscale for a readable image.
    px_per_frame = 3
    px_per_bin = 2
    W = nfreq * px_per_bin
    H = ntime * px_per_frame
    img = np.repeat(np.repeat(spec, px_per_frame, axis=0), px_per_bin, axis=1)
    img = (img * 255).astype(np.uint8)

    # Jet-ish colormap (blue->green->yellow->red) for artifact visibility.
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    v = img.astype(np.float32) / 255.0
    # simple 4-stop gradient
    stops = np.array([[0, 0, 40], [0, 80, 200], [40, 200, 120],
                      [240, 230, 60], [220, 40, 30]], dtype=np.float32)
    pos = v * (len(stops) - 1)
    idx = np.clip(pos.astype(int), 0, len(stops) - 2)
    frac = (pos - idx)[:, :, None]
    col = stops[idx] * (1 - frac) + stops[idx + 1] * frac
    rgb = col.astype(np.uint8)

    im = Image.fromarray(rgb, mode="RGB")

    # Left label gutter for frequency ticks.
    gutter = 70
    canvas = Image.new("RGB", (gutter + W, H + 24), (255, 255, 255))
    canvas.paste(im, (gutter, 0))
    d = ImageDraw.Draw(canvas)
    # frequency gridlines + labels
    for hz in (0, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000):
        if hz > f[-1]:
            continue
        y = int((1.0 - hz / f[-1]) * H)
        d.line([(gutter, y), (gutter + W, y)], fill=(0, 0, 0, 40), width=1)
        d.text((4, y - 8), "%dk" % (hz // 1000) if hz >= 1000 else str(hz),
               fill=(0, 0, 0))
    # time labels along the bottom
    dur = t[-1] if len(t) > 1 else 0
    for i in range(0, int(dur) + 1):
        x = int(i / dur * W) if dur > 0 else 0
        d.text((gutter + x, H + 6), "%ds" % i, fill=(0, 0, 0))
    d.text((gutter, H + 8), "", fill=(0, 0, 0))

    canvas.save(path)
    return path, (W, H)


def main():
    args = [a for a in sys.argv[1:]]
    fmax = 8000
    nfft = 2048
    if "--fmax" in args:
        i = args.index("--fmax"); fmax = int(args[i + 1]); del args[i:i + 2]
    if "--nfft" in args:
        i = args.index("--nfft"); nfft = int(args[i + 1]); del args[i:i + 2]
    if not args:
        print(__doc__); sys.exit(2)
    inp = args[0]
    out = args[1] if len(args) > 1 else inp.rsplit(".", 1)[0] + "_spec.png"

    x, sr = load_wav(inp)
    f, t, mag = make_spectrogram(x, sr, nfft=nfft, fmax=fmax)
    path, (W, H) = render(f, t, mag, out, dbmin=-85.0)
    dur = (len(x) / sr)
    # quick numeric summary
    db = 20 * np.log10(np.abs(mag) + 1e-9)
    print("input     : %s  (%.2fs @ %d Hz)" % (inp, dur, sr))
    print("spectro   : %s  (%dx%d px)" % (path, W, H))
    print("freq range: 0 .. %.0f Hz" % f[-1])
    print("peak dB   : %.1f" % db.max())
    # dominant frequency per coarse time slice (rough F0 proxy)
    ntime = mag.shape[1]
    print("dominant freq over time (rough F0 / energy peak):")
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        col = int(frac * (ntime - 1))
        pk = f[int(np.argmax(mag[:, col]))]
        print("  t=%.2fs : %d Hz" % (t[col], pk))


if __name__ == "__main__":
    main()

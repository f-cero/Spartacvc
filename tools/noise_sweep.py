#!/usr/bin/env python3
"""Noise-scale sweep: test which VAE noise scale preserves content.
Uses the authoritative rvc_python model (net.flow + net.dec).
"""
import os, sys
import numpy as np
import torch

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.dirname(__file__))
from rvc_ref import load_infer_pack

def main():
    base = os.path.join(BASE, "C:/Users/fluffy/AppData/Roaming/Python/Python313/site-packages/rvc_python/lib/infer_pack")
    pth = os.path.join(BASE, "models/voices/nahida-jp.pth")
    base = "C:/Users/fluffy/AppData/Roaming/Python/Python313/site-packages/rvc_python/lib/infer_pack"
    mo = load_infer_pack(base)
    cpt = torch.load(pth, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    net = mo.SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False, version=cpt.get("version","v1"))
    net.load_state_dict(cpt["weight"], strict=False)
    net.eval()

    T = 410
    m_p = torch.from_numpy(np.fromfile("scratch/m_p.bin",dtype=np.float32).reshape(192,410)).float().unsqueeze(0)   # [1,192,T]
    logs_p = torch.from_numpy(np.fromfile("scratch/logs_p.bin",dtype=np.float32).reshape(192,410)).float().unsqueeze(0)
    f0 = torch.from_numpy(np.fromfile("scratch/f0_model.bin",dtype=np.float32)).unsqueeze(0)  # [1,T]
    ds = torch.LongTensor([0])
    g = net.emb_g(ds.unsqueeze(0)).transpose(1,2)  # [1, gin, 1]

    print("exp(logs_p) rms=%.4f  m_p rms=%.4f" % (float((torch.exp(logs_p)**2).mean().sqrt()), float((m_p**2).mean().sqrt())))
    torch.manual_seed(0)
    scales = [0.0, 0.1, 0.3, 0.5, 0.66666, 1.0]
    for s in scales:
        rnd = torch.randn(1,192,410)
        z_p = (m_p + torch.exp(logs_p)*rnd*s)
        with torch.no_grad():
            z = net.flow(z_p, torch.ones(1,1,410), g=g, reverse=True)
            o = net.dec((z)[:, :, :410], f0, g=g)
        a = o.squeeze().numpy().astype(np.float32)
        import wave
        an = np.clip(a*32767,-32768,32767).astype(np.int16)
        fn = "scratch/noise_s_%s.wav" % str(s).replace(".","_")
        with wave.open(fn,"wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000); wf.writeframes(an.tobytes())
        print("  s=%-8s rms=%.4f max=%.4f -> %s" % (s, float(np.sqrt((a**2).mean())), float(np.max(np.abs(a))), fn))

if __name__ == "__main__":
    main()

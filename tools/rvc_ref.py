#!/usr/bin/env python
"""Dev-time reference: run the real RVC model (rvc_python) in PyTorch on the
same inputs the C pipeline uses, and dump intermediates (m_p, logs_p, z, audio)
so we can compare against the C pipeline.

Usage:
  python tools/rvc_ref.py <nahida.pth> <emb_model.bin> <pitch.bin> <f0_model.bin> \
      [--out-dir scratch]
  # emb_model.bin: float32 [T,768]  (C dumps this)
  # pitch.bin:     int64  [T]
  # f0_model.bin:  float32 [T]
"""
import sys, os, types, importlib.util
import numpy as np
import torch

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def load_infer_pack(base):
    # Stub the package hierarchy so `from rvc_python.lib.infer_pack import X`
    # resolves WITHOUT triggering rvc_python/__init__.py (which needs av/faiss/...).
    for pkg in ["rvc_python", "rvc_python.lib", "rvc_python.lib.infer_pack"]:
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m
    parent = sys.modules["rvc_python.lib.infer_pack"]
    commons   = _load("rvc_python.lib.infer_pack.commons",   base + "/commons.py")
    transforms= _load("rvc_python.lib.infer_pack.transforms",base + "/transforms.py")
    modules   = _load("rvc_python.lib.infer_pack.modules",   base + "/modules.py")
    attentions= _load("rvc_python.lib.infer_pack.attentions",base + "/attentions.py")
    models    = _load("rvc_python.lib.infer_pack.models",    base + "/models.py")
    models_onnx=_load("rvc_python.lib.infer_pack.models_onnx", base + "/models_onnx.py")
    for m in (commons, transforms, modules, attentions, models, models_onnx):
        setattr(parent, m.__name__.split(".")[-1], m)
    return models_onnx

def main():
    pth   = sys.argv[1]
    emb_b = sys.argv[2]
    pit_b = sys.argv[3]
    f0_b  = sys.argv[4]
    outd  = "scratch"
    if "--out-dir" in sys.argv:
        outd = sys.argv[sys.argv.index("--out-dir") + 1]
    os.makedirs(outd, exist_ok=True)

    import site
    sp = site.getusersitepackages()
    base = os.path.join(sp, "rvc_python", "lib", "infer_pack")
    mo = load_infer_pack(base)

    cpt = torch.load(pth, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    version = cpt.get("version", "v1")
    net = mo.SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False, version=version)
    net.load_state_dict(cpt["weight"], strict=False)
    net.eval()

    T = np.fromfile(emb_b, dtype=np.float32).shape[0] // 768
    emb = np.fromfile(emb_b, dtype=np.float32)[:T*768].reshape(1, T, 768)
    pitch = np.fromfile(pit_b, dtype=np.int64)[:T]
    f0 = np.fromfile(f0_b, dtype=np.float32)[:T]

    phone = torch.tensor(emb, dtype=torch.float32)          # [1, T, 768]
    phone_lengths = torch.tensor([T]).long()                 # [1]
    pitch_t = torch.tensor(pitch, dtype=torch.long).unsqueeze(0)  # [1, T]
    f0_t = torch.tensor(f0, dtype=torch.float32).unsqueeze(0)     # [1, T]
    ds = torch.LongTensor([0])
    rnd = torch.rand(1, 192, T)   # same shape; content-independent for m_p/z comparison

    enc = net.enc_p  # TextEncoder768
    with torch.no_grad():
        # --- replicate encoder forward, dumping each step (to match C dumps) ---
        x = enc.emb_phone(phone)                    # [1, T, C]
        x = enc.emb_pitch(pitch_t) + x              # pitch_t [1, T]
        x = x * (enc.hidden_channels ** 0.5)
        x = torch.nn.functional.leaky_relu(x, 0.1)
        def dumpf(name, t):  # t [1,T,C] or [1,C,T] -> save as [C,T] ravel (C-major)
            a = t.detach().float().numpy()  # [1, X, Y]
            if a.shape[1] == T:  # [1, T, C] -> transpose to [1, C, T]
                a = a.transpose(0, 2, 1)
            a.reshape(192, T).astype(np.float32).tofile(os.path.join(outd, name + ".bin"))
        dumpf("ref_enc_x_emb", x)
        x = x.transpose(1, 2)                       # [1, C, T]
        x_mask = torch.ones(1, 1, T)                # all frames valid
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)  # [1,1,T,T]
        # dump layer-0 Q/K/V (pointwise convs) for C comparison
        a0 = enc.encoder.attn_layers[0]
        def dumpq(name, t):
            a = t.detach().float().numpy().reshape(192, T).astype(np.float32)
            a.tofile(os.path.join(outd, name + ".bin"))
        q0 = a0.conv_q(x); k0 = a0.conv_k(x)
        dumpq("ref_mha_Q", q0); dumpq("ref_mha_K", k0); dumpq("ref_mha_V", a0.conv_v(x))
        # head-0 pre-softmax logits + post-softmax P via reference's own functions
        import math
        sd = math.sqrt(96)
        qh = q0.view(1, 2, 96, T).transpose(2,3); kh = k0.view(1, 2, 96, T).transpose(2,3)
        used_k = a0._get_relative_embeddings(a0.emb_rel_k, T)
        scores = torch.matmul(qh / sd, kh.transpose(-2, -1))
        scores = scores + a0._relative_position_to_absolute_position(a0._matmul_with_relative_keys(qh / sd, used_k))
        p = torch.softmax(scores, dim=-1)
        scores[0].numpy().astype(np.float32).tofile(os.path.join(outd, "ref_mha_logits_pre.bin"))
        p[0].numpy().astype(np.float32).tofile(os.path.join(outd, "ref_mha_P.bin"))
        # pre-conv_o output via reference's attention() (returns output before conv_o)
        pre_co, _ = a0.attention(q0, k0, a0.conv_v(x))
        pre_co[0].numpy().reshape(192, T).astype(np.float32).tofile(os.path.join(outd, "ref_mha_ao.bin"))
        for i in range(len(enc.encoder.attn_layers)):
            y = enc.encoder.attn_layers[i](x, x, attn_mask)
            if i == 0:
                dumpf("ref_attn0", y)  # [1,C,T] -> [C,T] (dumpf handles layout)
            pre_ln1 = x + y
            if i == 0: dumpf("ref_L0_preln1", pre_ln1)
            x = enc.encoder.norm_layers_1[i](pre_ln1)
            if i == 0: dumpf("ref_L0_ln1", x)
            y = enc.encoder.ffn_layers[i](x, x_mask)
            if i == 0: dumpf("ref_L0_ffn", y)
            pre_ln2 = x + y
            if i == 0: dumpf("ref_L0_preln2", pre_ln2)
            x = enc.encoder.norm_layers_2[i](pre_ln2)
            dumpf("ref_enc_x_L%d" % i, x)
        x = enc.proj(x * x_mask)                    # [1, 2*FC, T]
        m_p, logs_p = torch.split(x, 192, dim=1)
        # --- rest of pipeline ---
        g = net.emb_g(ds.unsqueeze(0)).transpose(1, 2)  # [1, gin, 1]
        z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
        z = net.flow(z_p, x_mask, g=g, reverse=True)
        o = net.dec((z * x_mask)[:, :, :T], f0_t, g=g)

    def dump(name, t):
        a = t.detach().float().numpy().ravel().astype(np.float32)
        a.tofile(os.path.join(outd, name + ".bin"))
        print("  dumped %-10s rms=%.4f" % (name, float(np.sqrt((a**2).mean()))))

    print("=== PyTorch reference (T=%d) ===" % T)
    dump("ref_m_p",    m_p)
    dump("ref_logs_p", logs_p)
    dump("ref_z",      z)
    dump("ref_audio",  o)

    # Also save m_p as [T, C] to match C layout for direct comparison
    def effrank(x):
        x = x - x.mean(0)
        ev = (np.linalg.svd(x, compute_uv=False)**2); ev = ev/ev.sum()
        return (ev.sum()**2)/((ev**2).sum()+1e-12)
    mp = m_p[0].numpy()  # [C, T]
    print("  ref m_p effrank=%.2f (C m_p effrank was 1.44)" % effrank(mp.T))

if __name__ == "__main__":
    main()

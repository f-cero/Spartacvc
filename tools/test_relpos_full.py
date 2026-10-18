#!/usr/bin/env python
"""Verify the FULL relative attention (keys + values) formula so the C fix is exact.
Reference: attentions.MultiHeadAttention.attention().
Candidate C formula:
  scores[l1,l2] = q[l1].k[l2]/sd + (|l1-l2|<=W ? q[l1].relk[W+(l2-l1)]/sd : 0)
  out[l1] = sum_l2 p[l1,l2] v[l2] + sum_l2 p[l1,l2] relv[ idx ]
  where the relv index for (l1,l2) is W+(l2-l1)  [verify sign]
"""
import sys, os, types, importlib.util, math
import numpy as np, torch

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

import site
sp = site.getusersitepackages()
base = os.path.join(sp, "rvc_python", "lib", "infer_pack")
for pkg in ["rvc_python","rvc_python.lib","rvc_python.lib.infer_pack"]:
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
parent = sys.modules["rvc_python.lib.infer_pack"]
for f in ["commons","transforms","modules","attentions"]:
    mm = _load("rvc_python.lib.infer_pack."+f, base+"/"+f+".py"); setattr(parent, f, mm)
att = sys.modules["rvc_python.lib.infer_pack.attentions"]

torch.manual_seed(0)
W = 10; d = 96; L = 20; H = 2
attn = att.MultiHeadAttention(channels=192, out_channels=192, n_heads=H, window_size=W)
attn.eval()
emb_rel_k = attn.emb_rel_k  # [1, 2W+1, d]
emb_rel_v = attn.emb_rel_v  # [1, 2W+1, d]

x = torch.randn(1, 192, L)
with torch.no_grad():
    q = attn.conv_q(x); k = attn.conv_k(x); v = attn.conv_v(x)
    ref_out, ref_p = attn.attention(q, k, v, mask=None)  # [1,192,L], [1,H,L,L]

    # my formula
    sd = math.sqrt(d)
    qh = q.view(1, H, d, L).transpose(2,3)  # [1,H,L,d]
    kh = k.view(1, H, d, L).transpose(2,3)
    vh = v.view(1, H, d, L).transpose(2,3)
    P = torch.zeros(1, H, L, L)
    for h in range(H):
        for l1 in range(L):
            for l2 in range(L):
                s = float((qh[0,h,l1] * kh[0,h,l2] / sd).sum())
                if abs(l1-l2) <= W:
                    s += float((qh[0,h,l1] * emb_rel_k[0, W+(l2-l1)] / sd).sum())
                P[0,h,l1,l2] = s
    P = torch.softmax(P, dim=-1)
    OUT = torch.zeros(1, H, L, d)
    for h in range(H):
        for l1 in range(L):
            acc = torch.zeros(d)
            for l2 in range(L):
                acc = acc + P[0,h,l1,l2] * vh[0,h,l2]
                if abs(l1-l2) <= W:
                    acc = acc + P[0,h,l1,l2] * emb_rel_v[0, W+(l2-l1)]
            OUT[0,h,l1] = acc
    OUT = OUT.transpose(2,3).contiguous().view(1, 192, L)  # pre-conv_o, like ref_out
    d_out = (ref_out - OUT).abs().max().item()
    d_p = (ref_p - P).abs().max().item()
    print("max|p_attn ref - mine| = %.6f" % d_p)
    print("max|out ref - mine|    = %.6f" % d_out)
    print("MATCH" if d_out < 1e-3 and d_p < 1e-3 else "MISMATCH")

#!/usr/bin/env python
"""Find the exact emb_rel_v index mapping for the relative-values path."""
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
emb_rel_v = attn.emb_rel_v  # [1, 2W+1, d]

x = torch.randn(1, 192, L)
with torch.no_grad():
    q = attn.conv_q(x); k = attn.conv_k(x); v = attn.conv_v(x)
    ref_out, ref_p = attn.attention(q, k, v, mask=None)

    # reference relative values contribution
    rw = attn._absolute_position_to_relative_position(ref_p)  # [1,H,L,2L-1]
    used_v = attn._get_relative_embeddings(emb_rel_v, L)  # [1,2L-1,d]
    ref_relv_out = attn._matmul_with_relative_values(rw, used_v)  # [1,H,L,d]

    # Try to find: ref_relv_out[l1] = sum_l2 p[l1,l2] * emb_rel_v[ idx(l1,l2) ]
    # For each (l1, l2), the coefficient of emb_rel_v[idx] is p[l1,l2].
    # Let's figure out idx(l1,l2) by checking which emb_rel_v row is used.
    # Approach: set p to one-hot and see which emb_rel_v row appears.
    # Simpler: for a given l1, ref_relv_out[l1] = sum_j rw[l1,j] * used_v[j]
    #   and used_v[j] = emb_rel_v[j - pad] for j in [pad, pad+2W+1)
    #   So ref_relv_out[l1] = sum_j rw[l1,j] * emb_rel_v[j-pad]
    # We want to express as sum_l2 p[l1,l2] * emb_rel_v[idx(l1,l2)]
    #   => the coefficient of emb_rel_v[m] is sum_j rw[l1,j] * [j-pad == m]
    #   = rw[l1, m+pad]
    # So coefficient of emb_rel_v[m] for row l1 is rw[l1, m+pad].
    # And we need this to equal p[l1, l2] where idx(l1,l2)=m, i.e., m = W + (l2-l1) (guess)
    #   => rw[l1, m+pad] should equal p[l1, l2] for m = W+(l2-l1).
    pad = L - (W + 1)  # pad_length
    print("L=%d W=%d pad_length=%d" % (L, W, pad))
    ok = True
    for l1 in [0, 5, 10, 19]:
        for l2 in [0, 5, 10, 19]:
            m = W + (l2 - l1)
            if 0 <= m <= 2*W:
                coeff = rw[0, 0, l1, m + pad].item()
                target = ref_p[0, 0, l1, l2].item()
                match = abs(coeff - target) < 1e-5
                if not match: ok = False
                print("l1=%2d l2=%2d m=W+(l2-l1)=%3d  rw[l1,m+pad]=%.6f  p[l1,l2]=%.6f  %s"
                      % (l1, l2, m, coeff, target, "OK" if match else "XX"))
    print("ALL OK" if ok else "SOME MISMATCH (try other sign)")

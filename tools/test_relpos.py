#!/usr/bin/env python
"""Verify the relative-position attention formula so the C fix is exact.
Check: scores_local[l1,l2] == sum_d q[l1,d] * emb_rel_k[(l1-l2)+W, d] / sqrt(d)
where W = window_size, valid for |l1-l2| <= W.
"""
import sys, os, types, importlib.util
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
    mm = _load("rvc_python.lib.infer_pack."+f, base+"/"+f+".py")
    setattr(parent, f, mm)
att = sys.modules["rvc_python.lib.infer_pack.attentions"]
commons = sys.modules["rvc_python.lib.infer_pack.commons"]

torch.manual_seed(0)
W = 10; d = 96; L = 20
attn = att.MultiHeadAttention(channels=192, out_channels=192, n_heads=2, window_size=W)
attn.eval()
# heads_share=True -> emb_rel_k [1, 21, 96]
emb_rel_k = attn.emb_rel_k  # [1, 21, 96]

x = torch.randn(1, 192, L)
with torch.no_grad():
    q = attn.conv_q(x)  # [1, 192, L]
    # reference scores_local via the model's own functions
    used = attn._get_relative_embeddings(emb_rel_k, L)  # [1, 2L-1, 96]
    qh = q.view(1, 2, 96, L).transpose(2,3)  # [1,2,L,96]
    import math
    rel_logits = attn._matmul_with_relative_keys(qh / math.sqrt(96), used)  # [1,2,L,2L-1]
    ref_local = attn._relative_position_to_absolute_position(rel_logits)  # [1,2,L,L]

    # my formula: idx = W + (l2 - l1), valid for |l1-l2| <= W
    my = torch.zeros(1, 2, L, L)
    for h in range(2):
        for l1 in range(L):
            for l2 in range(L):
                rel = l1 - l2
                if abs(rel) <= W:
                    idx = W + (l2 - l1)   # = W - rel
                    my[0,h,l1,l2] = float((qh[0,h,l1] * emb_rel_k[0, idx] / math.sqrt(96)).sum())
    diff = (ref_local - my).abs().max().item()
    print("emb_rel_k shape:", tuple(emb_rel_k.shape))
    print("ref_local shape:", tuple(ref_local.shape))
    print("max|ref - my_formula| = %.6f" % diff)
    print("MATCH" if diff < 1e-4 else "MISMATCH")

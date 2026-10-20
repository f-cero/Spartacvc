#!/usr/bin/env python3
"""
spartacvc_gpu.py — GPU (DirectML) voice conversion.

Converts the .pth voice model to ONNX IN MEMORY (no disk file), then pipes
the ONNX bytes to spartacvc.exe's stdin. spartacvc loads it via ORT
CreateSessionFromArray with the DmlExecutionProvider and a user-selected
GPU adapter (--gpu <id>).

The pure-C CPU path (spartacvc --voice <v.pth>) is unchanged; this wrapper
is the GPU path. Everything downstream of the voice model (HuBERT, RMVPE,
f0, FAISS retrieval, WAV output) is done by spartacvc in C.

Usage:
  python tools/spartacvc_gpu.py <input.wav> --voice <v.pth> --hubert <h.onnx> \
      [--rmvpe <r.onnx>] [--gpu 0] [--pitch +2] [--out out.wav] \
      [--index <idx.bin>] [--index-rate 0.05] [--backend cuda|dml|vulkan|cpu|auto]

Notes:
  - Requires: torch, rvc_python (for SynthesizerTrnMsNSFsidM).
  - The ONNX is exported with the same contract as rvc_python's export_onnx:
      inputs : phone[1,T,768] f32, phone_lengths[1] i64, pitch[1,T] i64,
               pitchf[1,T] f32, ds[1] i64, rnd[1,192,T] f32
      output : audio[1,1,T*480] f32
  - --gpu <id> is the DXGI adapter index (0 = primary).
"""
import sys
import io
import os
import subprocess
import argparse
import types
import importlib.util

import torch


def _import_rvc_model_cls():
    """Import SynthesizerTrnMsNSFsidM without triggering rvc_python/__init__.py
    (which imports parselmouth and other heavy/optional deps)."""
    if "rvc_python" not in sys.modules:
        spec = importlib.util.find_spec("rvc_python")
        rp_dir = os.path.dirname(spec.origin)
        rp = types.ModuleType("rvc_python")
        rp.__path__ = [rp_dir]
        sys.modules["rvc_python"] = rp
    from rvc_python.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
    return SynthesizerTrnMsNSFsidM


SynthesizerTrnMsNSFsidM = _import_rvc_model_cls()


def export_onnx_bytes(pth_path):
    """Load the .pth and export to ONNX in memory. Returns the ONNX bytes."""
    cpt = torch.load(pth_path, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    T = 200  # dummy sequence length (dynamic axis on time)
    test_phone = torch.rand(1, T, vec_channels)
    test_phone_lengths = torch.tensor([T]).long()
    test_pitch = torch.randint(size=(1, T), low=5, high=255)
    test_pitchf = torch.rand(1, T)
    test_ds = torch.LongTensor([0])
    test_rnd = torch.rand(1, 192, T)

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False, version=cpt.get("version", "v1")
    )
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()

    buf = io.BytesIO()
    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        buf,
        dynamic_axes={"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2]},
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["audio"],
        dynamo=False,   # force legacy exporter (DML-compatible, matches rvc_python)
    )
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="spartacvc GPU (DirectML) voice conversion")
    ap.add_argument("input", help="input WAV")
    ap.add_argument("--voice", required=True, help="voice .pth model")
    ap.add_argument("--hubert", required=True, help="HuBERT .onnx")
    ap.add_argument("--rmvpe", default=None, help="RMVPE .onnx (optional)")
    ap.add_argument("--gpu", type=int, default=0, help="DML GPU adapter index (0=primary)")
    ap.add_argument("--backend", default="auto", choices=["cuda", "dml", "vulkan", "cpu", "auto"], help="execution provider")
    ap.add_argument("--pitch", type=int, default=0, help="pitch shift semitones")
    ap.add_argument("--out", default=None, help="output WAV (default: <input>_spartacvc.wav)")
    ap.add_argument("--index", default=None, help="FAISS index .bin (optional)")
    ap.add_argument("--index-rate", type=float, default=0.05, help="FAISS index rate")
    ap.add_argument("--spartacvc", default=None, help="path to spartacvc.exe")
    args = ap.parse_args()

    # Locate spartacvc.exe (same dir as the project root, i.e. parent of tools/)
    if args.spartacvc:
        spartacvc = args.spartacvc
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        spartacvc = os.path.join(os.path.dirname(here), "spartacvc.exe")
        if not os.path.exists(spartacvc):
            spartacvc = "spartacvc.exe"

    print(f"[gpu] exporting {args.voice} to ONNX in memory ...", file=sys.stderr)
    onnx_bytes = export_onnx_bytes(args.voice)
    print(f"[gpu] ONNX size: {len(onnx_bytes)/1e6:.1f} MB", file=sys.stderr)

    out = args.out or os.path.splitext(args.input)[0] + "_spartacvc.wav"
    cmd = [
        spartacvc, args.input,
        "--voice-stdin",
        "--hubert", args.hubert,
        "--gpu", str(args.gpu),
        "--backend", args.backend,
        "--out", out,
    ]
    if args.rmvpe:
        cmd += ["--rmvpe", args.rmvpe]
    if args.pitch:
        cmd += ["--pitch", str(args.pitch)]
    if args.index:
        cmd += ["--index", args.index, "--index-rate", str(args.index_rate)]

    print(f"[gpu] running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        proc.stdin.write(onnx_bytes)
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Dev-time utility: export an RVC .pth voice model to a single ONNX file
containing the full encoder + flow + decoder graph.

This ONNX file is what the C runtime loads and runs via ONNX Runtime
(CPU or DirectML EP).  The pure-C path in src/ remains the CPU fallback.

Usage:
    python tools/export_onnx.py <voice.pth> <out.onnx>

Inputs (per rvc_python export contract):
    phone         [1, T, 768]   float32   HuBERT embeddings (v2 / 768-dim)
    phone_lengths [1]           int64     == T
    pitch         [1, T]        int64     quantized f0 (1..255)
    pitchf        [1, T]        float32   f0 in Hz (NSF excitation)
    ds            [1]           int64     speaker id (0 for single-speaker)
    rnd           [1, 192, T]   float32   VAE noise (N(0,1) * 0.66666)
Output:
    audio         [1, 1, T*upp] float32
"""
import sys
import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    pth, out = sys.argv[1], sys.argv[2]

    from rvc_python.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    cpt = torch.load(pth, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    T = 200  # dummy; time axis is dynamic
    test_phone = torch.rand(1, T, vec_channels)
    test_phone_lengths = torch.tensor([T]).long()
    test_pitch = torch.randint(size=(1, T), low=5, high=255)
    test_pitchf = torch.rand(1, T) * 200 + 50  # plausible Hz range
    test_ds = torch.LongTensor([0])
    test_rnd = torch.rand(1, 192, T)

    net_g = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False,
                                    version=cpt.get("version", "v1"))
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()

    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]

    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        out,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    print(f"Exported {pth} -> {out} (vec_channels={vec_channels})")


if __name__ == "__main__":
    main()

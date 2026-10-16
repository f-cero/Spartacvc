#!/usr/bin/env python3
"""Export RVC .pth voice model to ONNX.

Usage: python export_pth.py input.pth output.onnx

One-time conversion. The resulting .onnx can be loaded by ONNX Runtime
without any Python dependency at inference time.
"""
import sys
import os

def export(pth_path: str, onnx_path: str):
    import torch

    # Add the RVC infer_pack to path
    rvc_lib = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "temp", "rvcwheel", "rvcx", "rvc_python", "lib", "infer_pack")
    if not os.path.exists(rvc_lib):
        # Try standard location
        rvc_lib = os.path.join(os.path.expanduser("~"), "rvc_python", "lib", "infer_pack")
    if not os.path.exists(rvc_lib):
        print("ERROR: Cannot find rvc_python lib. Set RVC_LIB env var.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(rvc_lib))
    # Also need the package path
    sys.path.insert(0, os.path.join(os.path.dirname(rvc_lib), "..", ".."))

    from rvc_python.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

    print(f"Loading {pth_path}...")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)

    cfg = list(ckpt["config"])
    cfg[-3] = ckpt["weight"]["emb_g.weight"].shape[0]
    version = ckpt.get("version", "v2")
    sr = cfg[-1]
    if isinstance(sr, str):
        sr = {"48k": 48000, "32k": 32000, "24k": 24000}.get(sr, 48000)

    print(f"  version={version}, sr={sr}")
    print(f"  upsample_rates={cfg[12]}, upsample_kernel_sizes={cfg[14]}")

    print("Building model...")
    model = SynthesizerTrnMsNSFsidM(*cfg, version=version, is_half=False)
    model.load_state_dict(ckpt["weight"], strict=False)
    model = model.float()
    model.eval()
    model.remove_weight_norm()

    # Dummy inputs for tracing
    T = 100
    phone = torch.randint(0, 192, (1, T, 768)).float()
    phone_lengths = torch.tensor([T])
    pitch = torch.randint(1, 50, (1, T)).long()
    nsff0 = pitch.float()
    g = torch.tensor([0])
    inter_channels = cfg[2]  # 192 for v2
    rnd = torch.randn(1, inter_channels, T)
    max_len = T

    # Test forward
    print("Testing forward pass...")
    with torch.no_grad():
        out = model(phone, phone_lengths, pitch, nsff0, g, rnd, max_len)
        print(f"  output shape: {out.shape}")

    # Export
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (phone, phone_lengths, pitch, nsff0, g, rnd, max_len),
        onnx_path,
        input_names=["phone", "phone_lengths", "pitch", "nsff0", "g", "rnd", "max_len"],
        output_names=["audio"],
        dynamic_axes={
            "phone": {1: "T"},
            "pitch": {1: "T"},
            "nsff0": {1: "T"},
            "rnd": {2: "T"},
            "audio": {2: "T"},
        },
        opset_version=16,
        do_constant_folding=True,
    )
    print("Done!")

    # Validate
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"ONNX checker: PASSED ({len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers)")
    except ImportError:
        print("(onnx package not available, skipping checker)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.pth output.onnx", file=sys.stderr)
        sys.exit(1)
    export(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""
pth2manifest.py — One-time .pth → Spacvc manifest converter.

Usage:
    python pth2manifest.py model.pth -o model.spacvc

Reads a RVC .pth file (ZIP container with data.pkl + data/N tensor files)
and produces a .spacvc directory:
    model.spacvc/
        manifest.bin    — binary header with tensor metadata
        data/0          — raw tensor data (copied from .pth)
        data/1
        ...

The manifest format (all little-endian):
    Magic:         "SPCV" (4 bytes)
    Version:       uint32 = 1
    num_tensors:   uint32
    num_config:    uint32  (18 for RVC v2)
    config:        int64[num_config]
    For each tensor:
        name_len:    uint32
        name:        char[name_len]
        ndim:        uint32
        shape:       int64[ndim]
        dtype:       uint32  (0=fp16, 1=fp32, 2=int32, 3=int64, 4=uint8, 5=bool)
        data_file:   uint32  (index into data/ directory)
        size_bytes:  uint64
"""
import struct, sys, os, shutil, zipfile, torch

DTYPE_MAP = {
    torch.float16: 0,
    torch.float32: 1,
    torch.int32:   2,
    torch.int64:   3,
    torch.uint8:   4,
    torch.bool:    5,
}

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} model.pth [-o output.spacvc]")
        sys.exit(1)

    pth_path = sys.argv[1]
    out_path = None
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        out_path = sys.argv[idx + 1]
    else:
        base = os.path.splitext(os.path.basename(pth_path))[0]
        out_path = base + ".spacvc"

    print(f"Loading {pth_path}...")
    sd = torch.load(pth_path, map_location="cpu", weights_only=False)

    weight = sd["weight"]
    config = sd["config"]
    print(f"  Tensors: {len(weight)}")
    print(f"  Config:  {len(config)} values")

    # Create output directory
    data_dir = os.path.join(out_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Collect unique data files needed
    # In .pth format, all tensor data is in one or more data files
    # We'll copy the entire data file(s)
    print(f"Extracting data files...")
    with zipfile.ZipFile(pth_path) as z:
        names = z.namelist()
        # Find the prefix (e.g., "nahida-jp_e102_s21114/")
        prefix = ""
        for n in names:
            if n.endswith("data.pkl"):
                prefix = n[:n.index("data.pkl")]
                break

        # Copy data files
        data_files = set()
        for n in names:
            if n.startswith(prefix + "data/") and n != prefix + "data.pkl":
                fname = n[len(prefix):]  # e.g., "data/0"
                data_files.add(fname)
                out_file = os.path.join(out_path, fname)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                with z.open(n) as src, open(out_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  Copied {fname} ({os.path.getsize(out_file):,} bytes)")

    # Build tensor list — data file index = position in weight dict
    tensors = []
    for idx, (name, tensor) in enumerate(weight.items()):
        t = tensor.detach().contiguous()
        dtype_code = DTYPE_MAP.get(t.dtype, 1)
        tensors.append({
            "name": name,
            "ndim": t.dim(),
            "shape": list(t.shape),
            "dtype": dtype_code,
            "data_file": idx,
            "size_bytes": t.numel() * t.element_size(),
        })

    # Write manifest
    manifest_path = os.path.join(out_path, "manifest.bin")
    with open(manifest_path, "wb") as f:
        f.write(b"SPCV")                          # magic
        f.write(struct.pack("<I", 1))             # version
        f.write(struct.pack("<I", len(tensors)))  # num_tensors
        f.write(struct.pack("<I", len(config)))   # num_config
        for v in config:
            if isinstance(v, (list, tuple)):
                f.write(struct.pack("<q", -1))    # marker for non-int
            else:
                f.write(struct.pack("<q", int(v)))
        for t in tensors:
            name_bytes = t["name"].encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", t["ndim"]))
            for d in t["shape"]:
                f.write(struct.pack("<q", d))
            f.write(struct.pack("<I", t["dtype"]))
            f.write(struct.pack("<I", t["data_file"]))
            f.write(struct.pack("<Q", t["size_bytes"]))

    print(f"\nDone! Output: {out_path}")
    print(f"  manifest.bin: {os.path.getsize(manifest_path):,} bytes")
    total_data = sum(os.path.getsize(os.path.join(out_path, "data", d))
                     for d in os.listdir(os.path.join(out_path, "data")))
    print(f"  data/: {total_data:,} bytes")

if __name__ == "__main__":
    main()

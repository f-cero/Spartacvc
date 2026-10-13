# Getting RVC models

Spartacvς does **not** ship voice models. They are third-party works — check each
model's own license before redistributing.

## Verified sources (as of project start)

| Source | What's in it | Notes |
| --- | --- | --- |
| [`ArkanDash/rvc-genshin-impact`](https://huggingface.co/ArkanDash/rvc-genshin-impact) | ~45 Genshin Impact JP voices, RVC v1 + v2 | Each zip = `.pth` + `.index`. The "young girl" community standard (Nahida, Klee, Furina, Lumine, Paimon…) |
| [`AIHeaven/rvc-models`](https://huggingface.co/AIHeaven/rvc-models) | "Haruka", "Hikari" — general anime-girl voices | RVC zips |

> Always confirm a repo is active and its files intact before depending on it.
> HF repos can be renamed or removed.

## Model layout Spartacvς expects

Each voice model has a **feature dimension** (256 or 768) that dictates which
HuBERT base to pair with. The voice graph declares it in its `phone` input shape.

```
models/
├── vec-256-layer-9.onnx    # HuBERT base → 256-dim features (pair with 256-dim voices)
├── vec-768-layer-12.onnx   # HuBERT base → 768-dim features (pair with 768-dim voices)
├── rmvpe.onnx              # f0 extractor (shared across all voices)
└── <voice-name>.onnx       # RVC v1/v2 converter graph (e.g. beyonce.onnx, ganyu.onnx)
```

## `.pth` → `.onnx` (one-off conversion)

Spartacvς runs ONNX, but the community distributes torch `.pth`. Converting is a
**one-time** step in a throwaway Python environment — it is *not* a runtime
dependency of Spartacvς.

Procedure (to be finalized in Phase 0):
1. Load the `.pth` with the RVC `VC` module class.
2. `torch.onnx.export` the feature + converter graphs with fixed input shapes.
3. Validate the ONNX output against a torch render of the same input
   (spectral distance must be negligible).
4. Drop the resulting `.onnx` into `models/<voice-name>/`.

## Base graphs (HuBERT, RMVPE)

- **RMVPE**: official ONNX ships with RVC-WebUI (`rmvpe.onnx`).
- **HuBERT base**: pre-exported ONNX exists in the RVC ecosystem; otherwise
  export `hubert_base` once with the same throwaway env.

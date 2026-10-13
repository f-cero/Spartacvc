# Getting RVC models

Spartacvc does **not** ship voice models. They are third-party works — check each
model's own license before redistributing.

## Verified sources (as of project start)

| Source | What's in it | Notes |
| --- | --- | --- |
| [`ArkanDash/rvc-genshin-impact`](https://huggingface.co/ArkanDash/rvc-genshin-impact) | ~45 Genshin Impact JP voices, RVC v1 + v2 | Each zip = `.pth` + `.index`. The "young girl" community standard (Nahida, Klee, Furina, Lumine, Paimon…) |
| [`AIHeaven/rvc-models`](https://huggingface.co/AIHeaven/rvc-models) | "Haruka", "Hikari" — general anime-girl voices | RVC zips |

> Always confirm a repo is active and its files intact before depending on it.
> HF repos can be renamed or removed.

## Model layout Spartacvc expects

```
models/
├── hubert_base.onnx        # HuBERT base feature extractor (shared)
├── rmvpe.onnx              # f0 extractor (shared)
└── <voice-name>/
    ├── voice.onnx          # RVC v1/v2 converter graph
    ├── voice.index         # (optional) retrieval index
    └── model.json          # (optional) per-model defaults
```

## `.pth` → `.onnx` (one-off conversion)

Spartacvc runs ONNX, but the community distributes torch `.pth`. Converting is a
**one-time** step in a throwaway Python environment — it is *not* a runtime
dependency of Spartacvc.

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

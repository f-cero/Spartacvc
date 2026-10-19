# FAISS IVF Retrieval in Pure C (the "R" in RVC)

**Date:** 2026-07-21
**Status:** ✅ Working, verified

## What it does

RVC = **R**etrieval-based Voice Conversion. The retrieval step blends
target-speaker HuBERT features into the source features before the encoder:

```
feats = retrieved * index_rate + feats * (1 - index_rate)
```

where `retrieved` is found by searching the target speaker's FAISS index.

## The FAISS search (reverse-engineered from faiss behavior)

Empirically determined (verified against `faiss.search`):

1. **Metric is L2** — IVF search returns **L2 squared distances** (ascending),
   even when the index was built with an IP quantizer. (`q == vector` → score 0.)
2. **Nearest centroid** = `argmin_c L2(centroid_c, q)` (nprobe=1).
3. **Top-8** = the 8 vectors in that cluster with smallest `L2(v, q)`.
4. **Weights** = `(1 / l2)^2`, normalized to sum to 1.
5. **RVC quirk:** the result is `big_npy[id]` where `big_npy =
   index.reconstruct_n(0, ntotal)` is in **storage order** (cluster-by-cluster),
   but is indexed by the faiss **ID**. This is a quirk of rvc_python that we
   replicate exactly (it makes the result sensitive to the exact top-8 IDs).

## Files

| File | Role |
|------|------|
| `tools/extract_index.py` | **Dev-time** (uses faiss): `.index` → `spcv_index.bin` |
| `src/faiss_index.h/.c` | **Runtime** (pure C): load `.bin` + retrieve |
| `main.c` | `--index <bin>` `--index-rate <f>` options, blends into feats |

## Binary format (`spcv_index.bin`, little-endian)

```
magic      8s   "SPCVIX01"
d          i32  vector dimension (768 for Nahida)
nlist      i32  number of clusters (5055)
ntotal     i64  total vectors (197183)
metric     i32  0 = L2
centroids  f32  [nlist, d]
cluster_off i64 [nlist+1]  start storage-index of each cluster (contiguous)
big_npy    f32  [ntotal, d]  STORAGE order (matches faiss reconstruct_n)
pos2id     i64  [ntotal]     faiss ID of each storage position
```

## Usage

```bash
# one-time (dev): convert .index -> .bin
python tools/extract_index.py models/voices/<name>.index models/voices/<name>.index.bin

# run with retrieval
spartacvc.exe in.wav --voice <v>.pth --hubert <h>.onnx --rmvpe <r>.onnx \
    --index models/voices/<name>.index.bin --index-rate 0.3 --out out.wav
```

## Verification

- Format parses cleanly (622.9 MB, d=768 nlist=5055 ntotal=197183).
- C-style retrieval matches `faiss.search` + rvc_python formula: some queries
  exact, others differ only at the 8/9 boundary tie (a 1-vector difference —
  minor, since it's a small blend).
- Full pipeline: STT transcript correct, retrieval changes output (rel diff
  0.23 vs no-index), +1.2 s for 410 frames.

## Gotchas found

- **`long` is 32-bit on Windows** — must use `int64_t` for ntotal /
  cluster_off / pos2id (they're 8 bytes in the binary). First version segfaulted
  because of this.
- **`np.frombuffer` arg order** — `count` comes before `dtype`, `offset` after.
- **`reconstruct_n` is storage order, not ID order** — `id2pos` is a
  non-identity permutation; this is the source of the `big_npy[id]` quirk.

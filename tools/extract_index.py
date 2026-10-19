#!/usr/bin/env python3
"""Convert a FAISS .index (IVFFlat) into a C-friendly binary (spcv_index.bin).

Dev-time tool (uses faiss). The C runtime reads spcv_index.bin directly — no faiss needed.

Layout (little-endian):
  magic      : 8s  "SPCVIX01"
  d          : int32   vector dimension
  nlist      : int32   number of clusters
  ntotal     : int64   total vectors
  metric     : int32   0 = L2 (IVF search returns L2 distances)
  centroids  : float32 [nlist, d]   (row-major)
  cluster_off: int64   [nlist+1]    start storage-index of each cluster (contiguous)
  big_npy    : float32 [ntotal, d]  (STORAGE order — matches faiss reconstruct_n)
  pos2id     : int64   [ntotal]     original faiss ID of each storage position

C retrieval (replicates rvc_python exactly):
  1. c  = argmin over centroids of L2(centroid, q)
  2. in cluster c, take the 8 vectors with smallest L2(v, q)  -> (id, l2)
  3. w  = (1/l2)^2, normalized
  4. out = sum_i big_npy[id_i] * w_i     (big_npy indexed by ID, as rvc does)
"""
import sys, struct, os, numpy as np, faiss

def main():
    if len(sys.argv) < 3:
        print("usage: extract_index.py <in.index> <out.bin>"); return 1
    src, dst = sys.argv[1], sys.argv[2]
    idx = faiss.read_index(src)
    d = idx.d
    nlist = idx.nlist
    ntotal = idx.ntotal
    inv = idx.invlists

    cent = np.array(idx.quantizer.reconstruct_n(0, nlist), dtype=np.float32).reshape(nlist, d)
    sizes = np.array([inv.list_size(i) for i in range(nlist)], dtype=np.int64)
    cluster_off = np.zeros(nlist + 1, dtype=np.int64)
    np.cumsum(sizes, out=cluster_off[1:])
    assert cluster_off[-1] == ntotal, f"cluster total {cluster_off[-1]} != ntotal {ntotal}"

    # full storage-order array (single call)
    big = np.array(idx.reconstruct_n(0, ntotal), dtype=np.float32).reshape(ntotal, d)

    # verify storage is cluster-contiguous: cluster i's IDs should be big[cluster_off[i]:+n_i]
    pos2id = np.zeros(ntotal, dtype=np.int64)
    ok = True
    for i in range(nlist):
        n = sizes[i]
        if n == 0:
            continue
        ids = np.array([inv.get_single_id(i, j) for j in range(n)], dtype=np.int64)
        pos2id[cluster_off[i]:cluster_off[i] + n] = ids
    print(f"[extract_index] d={d} nlist={nlist} ntotal={ntotal}")

    with open(dst, "wb") as f:
        f.write(b"SPCVIX01")
        f.write(struct.pack("<ii", d, nlist))
        f.write(struct.pack("<q", ntotal))
        f.write(struct.pack("<i", 0))
        f.write(cent.tobytes())
        f.write(cluster_off.tobytes())
        f.write(big.tobytes())
        f.write(pos2id.tobytes())

    print(f"[extract_index] wrote {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
    return 0

if __name__ == "__main__":
    sys.exit(main())

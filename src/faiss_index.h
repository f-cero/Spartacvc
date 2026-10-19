/* spartacvc - FAISS IVF index retrieval (pure C, no faiss at runtime).
 *
 * Reads a C-friendly binary (tools/extract_index.py) and replicates rvc_python's
 * retrieval exactly:
 *   1. c   = argmin_c L2(centroid_c, q)
 *   2. in cluster c, take the 8 vectors with smallest L2(v, q)  -> (id, l2)
 *   3. w   = (1/l2)^2, normalized
 *   4. out = sum_i big_npy[id_i] * w_i     (big_npy indexed by ID, as rvc does)
 */
#ifndef SPCV_FAISS_INDEX_H
#define SPCV_FAISS_INDEX_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int       d;          /* vector dimension */
    int       nlist;      /* number of clusters */
    int64_t   ntotal;     /* total vectors */
    int       metric;     /* 0 = L2 */
    float    *centroids;  /* [nlist, d] row-major */
    int64_t  *cluster_off;/* [nlist+1] */
    float    *big_npy;    /* [ntotal, d] storage order */
    int64_t  *pos2id;     /* [ntotal] */
} SpcvIndex;

/* Load from a spcv_index.bin file. Returns 0 on success, -1 on error. */
int  spcv_index_load(const char* path, SpcvIndex* out);
void spcv_index_free(SpcvIndex* idx);

/* Retrieve: q [d] -> out [d]. Replicates rvc_python's index retrieval. */
void spcv_index_retrieve(const SpcvIndex* idx, const float* q, float* out);

#endif

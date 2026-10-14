// manifest.h — Spacvc manifest binary format reader
#ifndef MANIFEST_H
#define MANIFEST_H

#include <stdint.h>
#include <stddef.h>

#define MANIFEST_MAGIC    0x56435053  // "SPCV" little-endian
#define MANIFEST_VERSION  1
#define MAX_TENSORS       4096
#define MAX_NDIM          8
#define MAX_NAME_LEN      256

typedef struct {
    char     name[MAX_NAME_LEN];
    int      ndim;
    int64_t  shape[MAX_NDIM];
    int      dtype;       // 0=fp16, 1=fp32, 2=int32, 3=int64, 4=uint8, 5=bool
    int      data_file;   // index into data/ directory
    uint64_t size_bytes;
} manifest_tensor_t;

typedef struct {
    int64_t values[32];
    int     count;
} manifest_config_t;

typedef struct {
    manifest_tensor_t tensors[MAX_TENSORS];
    int               n_tensors;
    manifest_config_t config;
    char              base_dir[512];  // path to .spacvc directory
} manifest_t;

// Load manifest from a .spacvc directory
// Returns 0 on success, -1 on error
int manifest_load(manifest_t *m, const char *spacvc_dir);

// Load tensor data into buf (caller allocates size_bytes bytes)
// Returns 0 on success, -1 on error
int manifest_load_tensor(const manifest_t *m, int idx, void *buf);

// Get element size for a dtype
int manifest_dtype_size(int dtype);

#endif

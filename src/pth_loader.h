// pth_loader.h — load .pth files (ZIP + pickle) into C structures
#ifndef SPARTACVC_PTH_LOADER_H
#define SPARTACVC_PTH_LOADER_H

#include "pickle.h"
#include "zip_reader.h"

typedef struct {
    zip_archive_t  zip;
    pickle_ctx_t   pkl;
    int            owns_zip;
    // Resolved data file entries (data/N → zip_entry_t*)
    const zip_entry_t *data_files[1024];
    int            n_data_files;
} pth_model_t;

// Load a .pth file. Returns 0 on success, -1 on error.
int  pth_load(pth_model_t *m, const char *path);
void pth_free(pth_model_t *m);

// Read tensor data as raw bytes. Returns pointer into buf (caller allocates).
// dtype is converted to float32 if source is fp16.
int  pth_read_tensor_f32(const pth_model_t *m, const pth_tensor_t *t, float *out);
int  pth_read_tensor_raw(const pth_model_t *m, const pth_tensor_t *t, uint8_t *out);

// Find tensor by name. Returns pointer or NULL.
const pth_tensor_t *pth_find_tensor(const pth_model_t *m, const char *name);

// Print summary
void pth_print_summary(const pth_model_t *m);

#endif

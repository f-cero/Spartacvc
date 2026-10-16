// zip_reader.h — minimal ZIP archive reader (stored entries only)
#ifndef SPARTACVC_ZIP_READER_H
#define SPARTACVC_ZIP_READER_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#define ZIP_MAX_ENTRIES 1024

typedef struct {
    char     name[512];
    uint64_t offset;       // file data offset (after local header)
    uint32_t size;         // uncompressed size
    uint32_t compress_type;
} zip_entry_t;

typedef struct {
    FILE         *fh;
    zip_entry_t   entries[ZIP_MAX_ENTRIES];
    int           count;
} zip_archive_t;

// Open a ZIP archive. Returns 0 on success, -1 on error.
int  zip_open(zip_archive_t *za, const char *path);
void zip_close(zip_archive_t *za);

// Find entry by name (exact match). Returns pointer or NULL.
const zip_entry_t *zip_find(const zip_archive_t *za, const char *name);

// Find entry by prefix (e.g. "data/" matches "prefix/data/0").
// Returns pointer or NULL.
const zip_entry_t *zip_find_prefix(const zip_archive_t *za, const char *prefix);

// Read `len` bytes from entry into `buf`. Returns 0 on success.
int zip_read(const zip_archive_t *za, const zip_entry_t *entry,
             void *buf, size_t len);

// Read entire entry into a malloc'd buffer. Caller frees.
// Returns pointer or NULL.
uint8_t *zip_read_all(const zip_archive_t *za, const zip_entry_t *entry);

#endif // SPARTACVC_ZIP_READER_H

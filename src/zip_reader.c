// zip_reader.c — minimal ZIP archive reader using central directory
#include "zip_reader.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ZIP_LOCAL_SIG   0x04034B50
#define ZIP_CENTRAL_SIG 0x02014B50
#define ZIP_EOCD_SIG    0x06054B50

static uint16_t rd_u16(const uint8_t *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}
static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int seek64(FILE *f, int64_t off, int whence) {
#ifdef _WIN32
    return _fseeki64(f, (__int64)off, whence);
#else
    return fseeko(f, (off_t)off, whence);
#endif
}

static int64_t tell64(FILE *f) {
#ifdef _WIN32
    return (int64_t)_ftelli64(f);
#else
    return (int64_t)ftello(f);
#endif
}

int zip_open(zip_archive_t *za, const char *path) {
    memset(za, 0, sizeof(*za));
    za->fh = fopen(path, "rb");
    if (!za->fh) return -1;

    // Get file size
    seek64(za->fh, 0, SEEK_END);
    int64_t fsize = tell64(za->fh);
    if (fsize < 22) { fclose(za->fh); za->fh = NULL; return -1; }

    // Scan backwards for EOCD (max comment = 65535 bytes)
    int64_t scan_start = fsize - 22;
    if (scan_start < fsize - 22 - 65535) scan_start = fsize - 22 - 65535;
    if (scan_start < 0) scan_start = 0;

    int64_t eocd_offset = -1;
    uint8_t buf[32768]; // scan chunk
    int64_t scan_len = fsize - scan_start;
    if (scan_len > (int64_t)sizeof(buf)) scan_len = sizeof(buf);
    
    seek64(za->fh, scan_start, SEEK_SET);
    int64_t nread = fread(buf, 1, scan_len, za->fh);
    
    // Search for PK\x05\x06 in the buffer
    for (int64_t i = nread - 22; i >= 0; i--) {
        if (buf[i] == 'P' && buf[i+1] == 'K' && buf[i+2] == 0x05 && buf[i+3] == 0x06) {
            eocd_offset = scan_start + i;
            break;
        }
    }
    
    if (eocd_offset < 0) {
        fclose(za->fh); za->fh = NULL;
        return -1;
    }

    // Read EOCD
    uint8_t eocd[22];
    seek64(za->fh, eocd_offset, SEEK_SET);
    if (fread(eocd, 1, 22, za->fh) != 22) { fclose(za->fh); za->fh = NULL; return -1; }

    uint16_t cd_entries = rd_u16(eocd + 10);
    uint32_t cd_offset = rd_u32(eocd + 16);

    // Walk central directory entries (variable length due to name/extra/comment)
    int64_t pos = cd_offset;
    za->count = 0;
    for (int i = 0; i < cd_entries && za->count < ZIP_MAX_ENTRIES; i++) {
        uint8_t hdr[46];
        seek64(za->fh, pos, SEEK_SET);
        if (fread(hdr, 1, 46, za->fh) != 46) break;
        if (rd_u32(hdr) != ZIP_CENTRAL_SIG) break;

        uint16_t name_len = rd_u16(hdr + 28);
        uint16_t extra_len = rd_u16(hdr + 30);
        uint16_t comment_len = rd_u16(hdr + 32);
        uint32_t local_offset = rd_u32(hdr + 42);
        uint32_t uncomp_size = rd_u32(hdr + 24);
        uint16_t comp_type = rd_u16(hdr + 10);

        // Read filename
        zip_entry_t *e = &za->entries[za->count];
        memset(e, 0, sizeof(*e));
        e->compress_type = comp_type;
        e->size = uncomp_size;

        int rn = name_len < 511 ? name_len : 511;
        if (rn > 0) {
            if (fread(e->name, 1, rn, za->fh) != rn) break;
            e->name[rn] = '\0';
        }
        // Skip extra + comment
        seek64(za->fh, (int64_t)(extra_len + comment_len), SEEK_CUR);

        // Now read the local file header to get the actual data offset
        // (local header may have different name/extra lengths)
        uint8_t lh[30];
        seek64(za->fh, local_offset, SEEK_SET);
        if (fread(lh, 1, 30, za->fh) != 30) break;
        if (rd_u32(lh) != ZIP_LOCAL_SIG) break;

        uint16_t lh_name_len = rd_u16(lh + 26);
        uint16_t lh_extra_len = rd_u16(lh + 28);
        e->offset = (uint64_t)(local_offset + 30 + lh_name_len + lh_extra_len);

        za->count++;
        pos += 46 + name_len + extra_len + comment_len;
    }

    if (za->count == 0) {
        fclose(za->fh); za->fh = NULL;
        return -1;
    }
    return 0;
}

void zip_close(zip_archive_t *za) {
    if (za->fh) { fclose(za->fh); za->fh = NULL; }
    za->count = 0;
}

const zip_entry_t *zip_find(const zip_archive_t *za, const char *name) {
    for (int i = 0; i < za->count; i++) {
        if (strcmp(za->entries[i].name, name) == 0)
            return &za->entries[i];
    }
    return NULL;
}

const zip_entry_t *zip_find_prefix(const zip_archive_t *za, const char *prefix) {
    size_t plen = strlen(prefix);
    for (int i = 0; i < za->count; i++) {
        if (strncmp(za->entries[i].name, prefix, plen) == 0)
            return &za->entries[i];
    }
    return NULL;
}

int zip_read(const zip_archive_t *za, const zip_entry_t *entry,
             void *buf, size_t len) {
    if (len > entry->size) len = entry->size;
    if (seek64(za->fh, (int64_t)entry->offset, SEEK_SET) != 0) return -1;
    if (fread(buf, 1, len, za->fh) != len) return -1;
    return 0;
}

uint8_t *zip_read_all(const zip_archive_t *za, const zip_entry_t *entry) {
    uint8_t *buf = (uint8_t *)malloc(entry->size);
    if (!buf) return NULL;
    if (zip_read(za, entry, buf, entry->size) != 0) { free(buf); return NULL; }
    return buf;
}

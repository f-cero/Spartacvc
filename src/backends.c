/**
 * @file    backends.c
 * @brief   Enumerate ALL available inference backends + devices for `--gpu list`.
 *
 * Lists:
 *   - CUDA  (NVIDIA, via cudart dynamic load)  -> --backend cuda --gpu <ordinal>
 *   - DML   (DXGI adapters, via gpu_list.c)    -> --backend dml  --gpu <adapter>
 *   - CPU   (always available)                 -> --backend cpu
 *
 * Pure C: CUDA runtime (cudart64_*.dll) is loaded dynamically so the tool
 * reports "CUDA: not installed" cleanly when the runtime is absent.
 */
#define WIN32_LEAN_AND_MEAN
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef int  (WINAPI *PFN_cudaGetDeviceCount)(int *);
static PFN_cudaGetDeviceCount g_cudaGetDeviceCount = NULL;
static HMODULE g_cudart = NULL;

/* Try common cudart filenames (CUDA 11.x / 12.x). */
static int cudart_load(void) {
    if (g_cudart) return 1;
    const char *names[] = {
        "cudart64_110.dll", "cudart64_11.dll",
        "cudart64_12.dll", "cudart64_13.dll",
        "cudart64_101.dll", "cudart64_100.dll",
    };
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        g_cudart = LoadLibraryA(names[i]);
        if (g_cudart) break;
    }
    if (!g_cudart) return 0;
    g_cudaGetDeviceCount = (PFN_cudaGetDeviceCount)(void *)
        GetProcAddress(g_cudart, "cudaGetDeviceCount");
    return g_cudaGetDeviceCount != NULL;
}

static int spcv_list_cuda(int *out_count) {
    *out_count = 0;
    if (!cudart_load()) {
        printf("CUDA   : runtime DLL not found (NVIDIA driver / CUDA runtime missing)\n");
        return -1;
    }
    int count = 0;
    if (g_cudaGetDeviceCount(&count) != 0 || count <= 0) {
        printf("CUDA   : no CUDA-capable device (driver present, no GPU)\n");
        return 0;
    }
    printf("CUDA   : %d device(s)   --backend cuda\n", count);
    /* Use nvidia-smi (ships with every NVIDIA driver) for reliable details. */
    FILE *p = _popen(
        "nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>nul",
        "r");
    if (p) {
        char line[512];
        while (fgets(line, sizeof(line), p)) {
            line[strcspn(line, "\r\n")] = 0;
            if (!line[0]) continue;
            int idx = 0, cc_major = 0, cc_minor = 0;
            char name[256] = {0};
            long long mem_mib = 0;
            /* Split CSV into up to 4 fields (name has no commas). */
            char *f[4] = {0}; int nf = 0;
            char *s = line;
            for (int i = 0; i < 4 && s; i++) {
                f[i] = s;
                char *c = strchr(s, ',');
                if (c) { *c = 0; s = c + 1; } else s = NULL;
                nf++;
            }
            if (nf >= 1) idx = atoi(f[0] + strspn(f[0], " "));
            if (nf >= 2) { f[1] += strspn(f[1], " "); snprintf(name, sizeof(name), "%s", f[1]); }
            if (nf >= 3) { f[2] += strspn(f[2], " "); mem_mib = atoll(f[2]); }
            if (nf >= 4) { f[3] += strspn(f[3], " "); sscanf(f[3], "%d.%d", &cc_major, &cc_minor); }
            printf("  CUDA %d: %s   [sm_%d%d, vram=%.1f GB]   ->  --gpu %d\n",
                   idx, name, cc_major, cc_minor,
                   (double)mem_mib / 1024.0, idx);
        }
        _pclose(p);
    }
    *out_count = count;
    return 0;
}

/* Forward decl from gpu_list.c (DML enumeration). */
int spcv_list_gpus(void);

/**
 * @brief  Print a full report of available backends + devices.
 *
 * @return 0 on success (even if some backends are unavailable), -1 on hard error.
 */
int spcv_list_backends(void) {
    printf("spartacvc — available backends and devices\n");
    printf("==========================================\n");

    int cuda_n = 0;
    spcv_list_cuda(&cuda_n);

    /* DML (DXGI adapters) — spcv_list_gpus prints its own header; we just call it. */
    printf("\n");
    spcv_list_gpus();

    printf("\nCPU    : always available   --backend cpu\n");

    printf("\nDefault (--backend auto) picks the first available: CUDA > DML > CPU.\n");
    return 0;
}

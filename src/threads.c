/**
 * @file    threads.c
 * @brief   Implementation of spcv_par_for (see threads.h).
 *
 * Design notes:
 *  - The work range [0,n) is split into contiguous, non-overlapping chunks,
 *    one per worker. Because chunks never overlap, workers write to disjoint
 *    regions of the output and no atomics or locks are needed.
 *  - The job descriptors live on the caller's stack (inside spcv_par_for) and
 *    stay valid until every thread has been joined, so passing their address
 *    to the workers is safe.
 *  - Any thread-creation failure degrades that chunk to a serial run, so a
 *    partially-failing thread pool still produces a correct (slower) result.
 */
#include "threads.h"
#include <stdlib.h>

#ifdef _WIN32
#  include <process.h>
#  include <windows.h>
#  define SPCV_THREAD_CALL __stdcall
#  define SPCV_THREAD_RET  unsigned
#else
#  include <pthread.h>
#  define SPCV_THREAD_CALL
#  define SPCV_THREAD_RET  void *
#endif

typedef struct {
    int begin, end;
    spcv_par_fn fn;
    void *ctx;
} spcv_job_t;

#ifdef _WIN32
static SPCV_THREAD_RET SPCV_THREAD_CALL spcv_worker(void *arg) {
    spcv_job_t *j = (spcv_job_t *)arg;
    j->fn(j->begin, j->end, j->ctx);
    return 0;
}
#else
static void *spcv_worker(void *arg) {
    spcv_job_t *j = (spcv_job_t *)arg;
    j->fn(j->begin, j->end, j->ctx);
    return NULL;
}
#endif

int spcv_nthreads(void) {
    int n = 0;
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    n = (int)si.dwNumberOfProcessors;
#else
    n = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (n < 1)  n = 1;
    if (n > 64) n = 64;
    return n;
}

void spcv_par_for(int n, spcv_par_fn fn, void *ctx) {
    if (n <= 0) return;

    /* Avoid thread-creation overhead for small work, and never use more
     * threads than there are useful chunks. */
    int nt = spcv_nthreads();
    int maxt = n / 1024;            /* ~1k elements per thread minimum */
    if (maxt < 1) maxt = 1;
    if (nt > maxt) nt = maxt;
    if (nt > n)    nt = n;
    if (nt <= 1)  { fn(0, n, ctx); return; }

    int base = n / nt, rem = n % nt;
    spcv_job_t jobs[64];
    void *handles[64];
    int idx = 0, pos = 0;

    for (int t = 0; t < nt; t++) {
        int chunk = base + (t < rem ? 1 : 0);
        if (chunk <= 0) continue;
        jobs[idx].begin = pos;
        jobs[idx].end   = pos + chunk;
        jobs[idx].fn    = fn;
        jobs[idx].ctx   = ctx;
        pos += chunk;

#ifdef _WIN32
        unsigned tid = 0;
        HANDLE h = (HANDLE)_beginthreadex(NULL, 0, spcv_worker, &jobs[idx], 0, &tid);
        if (!h) { fn(jobs[idx].begin, jobs[idx].end, ctx); idx++; continue; }
        handles[idx] = (void *)h;
#else
        pthread_t th;
        if (pthread_create(&th, NULL, spcv_worker, &jobs[idx]) != 0) {
            fn(jobs[idx].begin, jobs[idx].end, ctx); idx++; continue;
        }
        handles[idx] = (void *)&th;
#endif
        idx++;
    }

    for (int i = 0; i < idx; i++) {
#ifdef _WIN32
        HANDLE h = (HANDLE)handles[i];
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
#else
        pthread_t *th = (pthread_t *)handles[i];
        pthread_join(*th, NULL);
#endif
    }
}

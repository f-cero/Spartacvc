/**
 * @file    threads.h
 * @brief   Tiny parallel-for helper (Win32 / pthreads / serial fallback).
 *
 * spcv_par_for(n, fn, ctx) splits the half-open range [0, n) across worker
 * threads and calls fn(begin, end, ctx) on each. It is a barrier: it returns
 * only after every worker has finished. Small n (or thread-creation failure)
 * falls back to a serial call so callers never need to special-case it.
 */
#ifndef SPCV_THREADS_H
#define SPCV_THREADS_H

typedef void (*spcv_par_fn)(int begin, int end, void *ctx);

void spcv_par_for(int n, spcv_par_fn fn, void *ctx);
int  spcv_nthreads(void);

#endif /* SPCV_THREADS_H */

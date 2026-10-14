/**
 * @file    debug.h
 * @brief   Compile-time + runtime gated tracing for spartacvc.
 *
 * Two gates (BOTH must be true for traces to emit):
 *   1. Compile-time:  -DSPCV_DEBUG   (or CMake -DDEBUG=ON)
 *   2. Runtime:       --debug flag  OR  SPCV_DEBUG=1 env var
 *
 * In release builds (SPCV_DEBUG not defined), every SPCV_TRACE()
 * compiles to a no-op. Zero overhead.
 *
 * Trace output goes to:
 *   - stderr  (always, when active)
 *   - spcv_trace.log  (append mode, created in CWD)
 */
#ifndef SPCV_DEBUG_H
#define SPCV_DEBUG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* ------------------------------------------------------------------ */
/*  Runtime gate                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief  Set the runtime debug flag (call from main after arg parse).
 * @param  on  1 = enable, 0 = disable
 */
void spcv_debug_set(int on);

/**
 * @brief  Query the runtime debug flag.
 * @return 1 if debug is active, 0 otherwise
 */
int spcv_debug_active(void);

/* ------------------------------------------------------------------ */
/*  Trace macro                                                        */
/* ------------------------------------------------------------------ */

#ifdef SPCV_DEBUG

/**
 * @brief  Emit a trace line if runtime debug is enabled.
 *
 * @param  module  short module tag (e.g. "onnx", "io", "f0")
 * @param  fmt     printf-style format string
 * @param  ...     format args
 *
 * Output format:  [spcv] <module> <message>\n
 * Goes to stderr AND spcv_trace.log (append).
 */
#define SPCV_TRACE(module, ...) do {                                    \
    if (spcv_debug_active()) {                                          \
        fprintf(stderr, "[spcv] %s ", module);                          \
        fprintf(stderr, __VA_ARGS__);                                   \
        fprintf(stderr, "\n");                                          \
        fflush(stderr);                                                 \
        FILE *_f = fopen("spcv_trace.log", "a");                        \
        if (_f) {                                                       \
            fprintf(_f, "[spcv] %s ", module);                          \
            fprintf(_f, __VA_ARGS__);                                   \
            fprintf(_f, "\n");                                          \
            fclose(_f);                                                 \
        }                                                               \
    }                                                                   \
} while(0)

#else  /* !SPCV_DEBUG — release build: compile to nothing */

#define SPCV_TRACE(module, ...) do { (void)0; } while(0)

#endif /* SPCV_DEBUG */

#endif /* SPCV_DEBUG_H */

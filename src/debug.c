/**
 * @file    debug.c
 * @brief   Runtime debug gate implementation.
 */
#include "debug.h"

static int g_debug_active = 0;

void spcv_debug_set(int on) {
    g_debug_active = on ? 1 : 0;
}

int spcv_debug_active(void) {
    return g_debug_active;
}

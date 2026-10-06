/*
 * ps3recomp - cached environment lookups for hot paths
 *
 * The runtime's diagnostics are switched by environment variables, and a lot
 * of them are read where they are used: once per FIFO drain, per SPURS job,
 * per event-queue receive. getenv() is not free -- the Windows CRT takes a
 * lock and scans the environment block -- and on Drakengard 3 those lookups
 * were 7% of the render thread and 11% of the RSX walker (they show up as
 * strchr in a profile).
 *
 * ps3_env(name) answers from a table keyed on the POINTER of the name, which
 * for a string literal is stable, so a hit is a hash and a compare. The
 * value is read once and copied; a variable changed after its first lookup
 * keeps its old value here, which is what "set before launch" switches want.
 * A file opts in by defining getenv to it after its includes:
 *
 *     #include "ps3emu/env_cache.h"
 *     #define getenv(name) ps3_env(name)
 *
 * Code that must see a variable change at run time (the replay harness's
 * per-present dump path) simply does not opt in.
 */
#ifndef PS3EMU_ENV_CACHE_H
#define PS3EMU_ENV_CACHE_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PS3_ENV_CACHE_SLOTS 2048u

typedef struct ps3_env_slot {
    const char* key;    /* the name's address; NULL free, PS3_ENV_BUSY being filled */
    const char* val;    /* a private copy of the value, or NULL when unset */
} ps3_env_slot;

extern ps3_env_slot g_ps3_env_cache[PS3_ENV_CACHE_SLOTS];

/* Slow path: claim a slot, read the variable, publish. Out of line. */
const char* ps3_env_fill(const char* name, unsigned first_slot);

static inline const char* ps3_env(const char* name)
{
    const uint64_t h = ((uint64_t)(uintptr_t)name >> 2) * 0x9E3779B97F4A7C15ull;
    const unsigned first = (unsigned)(h >> 40) & (PS3_ENV_CACHE_SLOTS - 1u);
    for (unsigned k = 0; k < 4; k++) {
        const ps3_env_slot* s = &g_ps3_env_cache[(first + k) & (PS3_ENV_CACHE_SLOTS - 1u)];
        const char* key = __atomic_load_n(&s->key, __ATOMIC_ACQUIRE);
        if (key == name) return s->val;
        if (!key) break;
    }
    return ps3_env_fill(name, first);
}

#ifdef __cplusplus
}
#endif
#endif /* PS3EMU_ENV_CACHE_H */

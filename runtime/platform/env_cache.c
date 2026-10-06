/*
 * ps3recomp - cached environment lookups (see include/ps3emu/env_cache.h)
 */
#include "ps3emu/env_cache.h"

#include <string.h>

ps3_env_slot g_ps3_env_cache[PS3_ENV_CACHE_SLOTS];

/* A slot being filled: readers treat it as "not mine" and probe on. */
#define PS3_ENV_BUSY ((const char*)(uintptr_t)1)

const char* ps3_env_fill(const char* name, unsigned first_slot)
{
    /* The real getenv: this file does not define the macro. The value is
     * copied because the CRT may move its environment block on a later
     * putenv, and the cache hands the pointer out for the life of the
     * process. */
    const char* v = getenv(name);
    char* copy = NULL;
    if (v) {
        const size_t n = strlen(v) + 1;
        copy = (char*)malloc(n);
        if (copy) memcpy(copy, v, n);
    }
    for (unsigned k = 0; k < 16; k++) {
        ps3_env_slot* s = &g_ps3_env_cache[(first_slot + k) & (PS3_ENV_CACHE_SLOTS - 1u)];
        const char* key = __atomic_load_n(&s->key, __ATOMIC_ACQUIRE);
        if (key == name) { free(copy); return s->val; }     /* another thread got there first */
        if (key) continue;
        const char* expect = NULL;
        if (!__atomic_compare_exchange_n(&s->key, &expect, PS3_ENV_BUSY, 0,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            continue;                                        /* lost the slot; try the next */
        s->val = copy;
        __atomic_store_n(&s->key, name, __ATOMIC_RELEASE);
        return copy;
    }
    /* Neighbourhood full: answer without caching (the copy is the answer's
     * storage, so it stays). */
    return copy;
}
